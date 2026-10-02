from __future__ import annotations

import fasttext

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.inventory.static import StaticInventory
from ollama_runner.nlu.backend import LLMNLUBackend
from ollama_runner.pipeline import Pipeline
from ollama_runner.process.cache import PostgresCache
from ollama_runner.process.parser import OllamaParser
from ollama_runner.process.resolver import FastTextResolver
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.settings import get_settings
from ollama_runner.skills.book import Executor


_PARSERS: dict[str, OllamaParser] = {}
_SEMANTIC_PARSERS: dict[str, LLMNLUBackend] = {}
_CACHES: dict[str, PostgresCache] = {}
_RESOLVER: FastTextResolver | None = None
_REGISTRY: DeviceRegistry | None = None
_INVENTORY = StaticInventory()


def get_resolver() -> FastTextResolver:
    return _get_resolver()


def semantic_pipeline(*, model: str | None = None) -> SemanticPipeline:
    """Same Ollama model, semantic IR, capability resolver, existing sink."""

    settings = get_settings()
    chosen = model or settings.ollama_model
    registry = _get_registry()
    return SemanticPipeline(
        _get_semantic_parser(chosen),
        CapabilityResolver(registry),
        Executor(registry),
        registry,
    )


def default_pipeline(*, model: str | None = None, cache: bool = True) -> Pipeline:
    settings = get_settings()
    chosen = model or settings.ollama_model
    pipeline = Pipeline().or_else(_get_parser(chosen), _get_resolver())
    if cache:
        pipeline.via(_get_cache(chosen))
    return pipeline


def _get_parser(model: str) -> OllamaParser:
    parser = _PARSERS.get(model)
    if parser is not None:
        return parser

    settings = get_settings()
    parser = OllamaParser(
        model=model,
        prompt_path=settings.resolved_prompt_path,
        base_url=settings.ollama_url,
        temperature=settings.ollama_temperature,
        keep_alive=settings.ollama_keep_alive,
        timeout=settings.ollama_timeout,
    )
    _PARSERS[model] = parser
    return parser


def _get_registry() -> DeviceRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DeviceRegistry.from_static()
    return _REGISTRY


def _get_semantic_parser(model: str) -> LLMNLUBackend:
    parser = _SEMANTIC_PARSERS.get(model)
    if parser is not None:
        return parser

    settings = get_settings()
    parser = LLMNLUBackend(
        model=model,
        prompt_path=settings.resolved_prompt_path.parent / "semantic.md",
        base_url=settings.ollama_url,
        temperature=settings.ollama_temperature,
        keep_alive=settings.ollama_keep_alive,
        timeout=settings.ollama_timeout,
    )
    _SEMANTIC_PARSERS[model] = parser
    return parser


def _get_resolver() -> FastTextResolver:
    global _RESOLVER

    if _RESOLVER is not None:
        return _RESOLVER

    settings = get_settings()
    model = fasttext.load_model(str(settings.resolved_fasttext_path))
    _RESOLVER = FastTextResolver(model, _INVENTORY)
    return _RESOLVER


def _get_cache(model: str) -> PostgresCache:
    cache = _CACHES.get(model)
    if cache is not None:
        return cache

    settings = get_settings()
    cache = PostgresCache(
        settings.postgres_conninfo(),
        model=model,
        inventory_v=_INVENTORY.version,
    )
    _CACHES[model] = cache
    return cache


def close_resources() -> None:
    for parser in _PARSERS.values():
        parser.close()
    _PARSERS.clear()

    for parser in _SEMANTIC_PARSERS.values():
        parser.close()
    _SEMANTIC_PARSERS.clear()

    for cache in _CACHES.values():
        cache.close()
    _CACHES.clear()
