from __future__ import annotations

from typing import Mapping

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.backend import NLUBackend
from ollama_runner.nlu.slots import apply_explicit
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import DeviceRuntime, RequestContext
from ollama_runner.skills.book import Executor
from ollama_runner.types import Result


class SemanticPipeline:
    """text → NLU → SemanticCommand → resolver → skill → existing sink."""

    def __init__(
        self,
        nlu: NLUBackend,
        resolver: CapabilityResolver,
        executor: Executor,
        registry: DeviceRegistry,
    ) -> None:
        self._nlu = nlu
        self._resolver = resolver
        self._executor = executor
        self._registry = registry

    def run(
        self,
        text: str,
        sink,
        *,
        context: RequestContext | None = None,
        state: Mapping[str, DeviceRuntime] | None = None,
    ) -> Result:
        context = context or RequestContext()
        parsed = self._nlu.parse(text, context)
        semantic, slots, notes = apply_explicit(text, parsed, self._registry)
        resolved = self._resolver.resolve(
            semantic,
            context=context,
            state=state,
            text=text,
            slots=slots,
            nlu_intent=parsed.intent,
            normalization=notes,
        )
        return self._executor.execute(resolved, sink)

    def warmup(self) -> None:
        self._nlu.parse("прогрев модели", RequestContext())
