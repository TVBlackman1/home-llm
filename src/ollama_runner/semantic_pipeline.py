from __future__ import annotations

from typing import Mapping

from ollama_runner.nlu.backend import NLUBackend
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
    ) -> None:
        self._nlu = nlu
        self._resolver = resolver
        self._executor = executor

    def run(
        self,
        text: str,
        sink,
        *,
        context: RequestContext | None = None,
        state: Mapping[str, DeviceRuntime] | None = None,
    ) -> Result:
        context = context or RequestContext()
        semantic = self._nlu.parse(text, context)
        resolved = self._resolver.resolve(
            semantic,
            context=context,
            state=state,
            text=text,
        )
        return self._executor.execute(resolved, sink)

    def warmup(self) -> None:
        self._nlu.parse("прогрев модели", RequestContext())
