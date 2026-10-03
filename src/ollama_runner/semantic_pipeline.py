from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.backend import NLUBackend
from ollama_runner.nlu.parse import DeterministicParse, parse_deterministic
from ollama_runner.nlu.slots import apply_explicit
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import DeviceRuntime, ExplicitSlots, RequestContext, SemanticCommand
from ollama_runner.skills.book import Executor
from ollama_runner.types import Result


@dataclass(frozen=True)
class NLUTrace:
    text: str
    handled: bool
    reason: str
    evidence: tuple[str, ...]
    deterministic_intent: str | None
    llm_intent: str | None
    final_intent: str
    deterministic_s: float
    fallback_s: float
    total_s: float


class SemanticPipeline:
    """text → deterministic parse → Ministral only if intent is unknown → resolver."""

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
        self.traces: list[NLUTrace] = []

    def run(
        self,
        text: str,
        sink,
        *,
        context: RequestContext | None = None,
        state: Mapping[str, DeviceRuntime] | None = None,
    ) -> Result:
        started = time.perf_counter()
        context = context or RequestContext()
        parsed_at = time.perf_counter()
        deterministic = parse_deterministic(text, self._registry)
        deterministic_s = time.perf_counter() - parsed_at
        fallback_s = 0.0
        llm_intent: str | None = None

        if deterministic.handled:
            semantic = deterministic.command()
            slots = deterministic.slots
            notes = deterministic.evidence
            nlu_intent = deterministic.intent or ""
        else:
            fallback_at = time.perf_counter()
            parsed = self._nlu.parse(text, context)
            fallback_s = time.perf_counter() - fallback_at
            llm_intent = parsed.intent
            semantic, slots, notes = _merge(text, deterministic, parsed, self._registry)
            nlu_intent = parsed.intent

        resolved = self._resolver.resolve(
            semantic,
            context=context,
            state=state,
            text=text,
            slots=slots,
            nlu_intent=nlu_intent,
            normalization=notes,
        )
        result = self._executor.execute(resolved, sink)
        self.traces.append(
            NLUTrace(
                text=text,
                handled=deterministic.handled,
                reason=deterministic.reason,
                evidence=deterministic.evidence,
                deterministic_intent=deterministic.intent,
                llm_intent=llm_intent,
                final_intent=semantic.intent,
                deterministic_s=deterministic_s,
                fallback_s=fallback_s,
                total_s=time.perf_counter() - started,
            )
        )
        return result

    def warmup(self) -> None:
        """Load the model. The warmup phrase is not a benchmark request."""

        if parse_deterministic("прогрев модели", self._registry).handled:
            return
        self._nlu.parse("прогрев модели", RequestContext())


def _merge(
    text: str,
    deterministic: DeterministicParse,
    parsed: SemanticCommand,
    registry: DeviceRegistry,
) -> tuple[SemanticCommand, ExplicitSlots, tuple[str, ...]]:
    command, slots, notes = apply_explicit(text, parsed, registry)
    if not deterministic.value or command.arguments.get("value") == deterministic.value:
        return command, slots, notes
    arguments = dict(command.arguments)
    arguments["value"] = deterministic.value
    return (
        SemanticCommand(intent=command.intent, target=command.target, arguments=arguments),
        slots,
        notes + ("value from text",),
    )
