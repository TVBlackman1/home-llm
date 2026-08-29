from __future__ import annotations

from pathlib import Path
from typing import Self

from returns.maybe import Maybe
from returns.pipeline import flow

from ollama_runner.protocols import Cache, Parser, Resolver, Sink, Source, Transcriber
from ollama_runner.sources.text import FileSource, TextSource
from ollama_runner.sources.whisper import WhisperSource
from ollama_runner.types import Command, Result, Utterance


class Pipeline:
    """Fluent builder: source → cache fastpath | parse+resolve → sink."""

    def __init__(self) -> None:
        self._source: Source | None = None
        self._cache: Cache | None = None
        self._parser: Parser | None = None
        self._resolver: Resolver | None = None
        self._sink: Sink | None = None

    def from_text(self, text: str) -> Self:
        self._source = TextSource(text)
        return self

    def from_file(self, path: str | Path) -> Self:
        self._source = FileSource(path)
        return self

    def from_audio(
        self,
        path: str | Path,
        transcriber: Transcriber | None = None,
    ) -> Self:
        self._source = WhisperSource(path, client=transcriber)
        return self

    def via(self, cache: Cache) -> Self:
        self._cache = cache
        return self

    def or_else(self, parser: Parser, resolver: Resolver) -> Self:
        self._parser = parser
        self._resolver = resolver
        return self

    def into(self, sink: Sink) -> Self:
        self._sink = sink
        return self

    def invalidate(self, text: str) -> Self:
        if self._cache is not None:
            self._cache.invalidate(text)
        return self

    def warmup(self) -> Self:
        if self._parser is None:
            raise RuntimeError("pipeline parser is not set")
        self._parser.parse("прогрев модели")
        return self

    def process(self, utterance: Utterance | None = None) -> Command:
        if utterance is None:
            if self._source is None:
                raise RuntimeError("pipeline source is not set")
            utterance = self._source.read()

        if self._parser is None or self._resolver is None:
            raise RuntimeError("pipeline parser/resolver is not set")

        return flow(
            utterance,
            self._lookup,
            lambda cached: cached.or_else_call(
                lambda: self._compute_and_store(utterance)
            ),
        )

    def run(self) -> Result:
        if self._sink is None:
            raise RuntimeError("pipeline sink is not set")

        return self._sink.send(self.process())

    def _lookup(self, utterance: Utterance) -> Maybe[Command]:
        if self._cache is None:
            return Maybe.empty

        return Maybe.from_optional(self._cache.get(utterance.text))

    def _compute_and_store(self, utterance: Utterance) -> Command:
        command = flow(
            utterance.text,
            self._parser.parse,
            self._resolver.resolve,
        )

        if self._cache is not None:
            self._cache.put(utterance.text, command)

        return command
