from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.backend import NLUBackend
from ollama_runner.nlu.parse import DeterministicParse, parse_deterministic
from ollama_runner.nlu.slots import apply_explicit
from ollama_runner.nlu.structure import parse_plan
from ollama_runner.request import (
    RequestResult,
    command_from_result,
    emit_request,
    overall_status,
)
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import (
    CommandOutcome,
    DeviceRuntime,
    ExplicitSlots,
    RequestContext,
    SemanticCommand,
    SemanticNLUOutcome,
)
from ollama_runner.skills.book import Executor
from ollama_runner.types import Command, Result


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
        self.observations: list[dict] = []
        self._request_ids = 0
        self.last_request: RequestResult | None = None

    def run(
        self,
        text: str,
        sink,
        *,
        context: RequestContext | None = None,
        state: Mapping[str, DeviceRuntime] | None = None,
    ) -> Result:
        started = time.perf_counter()
        self._request_ids += 1
        request_id = self._request_ids
        context = context or RequestContext()
        parsed_at = time.perf_counter()
        plan = parse_plan(text, self._registry)
        deterministic_s = time.perf_counter() - parsed_at
        fallback_s = 0.0
        llm_intent: str | None = None
        self._resolve_log = []
        llm_raw = None
        final_command = None
        merge_notes: tuple[str, ...] = ()
        merge_ran = False
        llm_outcome = None
        discarded_intent = None
        records = []

        if plan.fully_parsed and plan.commands:
            result, records = _execute_plan(self, plan, sink, context, state, text)
            final_intent = plan.commands[-1].intent
            handled = True
            reason = plan.reason
            evidence = plan.evidence
            deterministic_intent = plan.commands[0].intent
            final_command = plan.commands[-1]
        else:
            # A partial plan is not executed. The unresolved span stays for a later policy.
            deterministic = parse_deterministic(text, self._registry)
            fallback_at = time.perf_counter()
            outcome = _as_outcome(self._nlu.parse(text, context))
            fallback_s = time.perf_counter() - fallback_at
            if isinstance(outcome, CommandOutcome):
                parsed = outcome.command
                llm_intent = parsed.intent
                llm_raw = parsed
                semantic, slots, notes = _merge(text, deterministic, parsed, self._registry)
                merge_notes = notes
                merge_ran = True
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
                self._resolve_log.append(f"{resolved.status}/{resolved.reason or ''}")
                result = self._executor.execute(resolved, sink)
                records.append(
                    _recorded(self, "llm", "command", semantic, resolved, result)
                )
                final_intent = semantic.intent
                final_command = semantic
            else:
                llm_intent = None
                result = _stopped(outcome.kind)
                records.append(
                    command_from_result(source="llm", decision=outcome.kind, result=result)
                )
                final_intent = ""
            handled = False
            reason = plan.reason
            evidence = plan.evidence
            deterministic_intent = None
            llm_outcome = outcome.kind
            discarded_intent = getattr(outcome, "discarded_intent", None)

        self.traces.append(
            NLUTrace(
                text=text,
                handled=handled,
                reason=reason,
                evidence=evidence,
                deterministic_intent=deterministic_intent,
                llm_intent=llm_intent,
                final_intent=final_intent,
                deterministic_s=deterministic_s,
                fallback_s=fallback_s,
                total_s=time.perf_counter() - started,
            )
        )
        self.observations.append(
            {
                "handled": handled,
                "reason": reason,
                "plan_commands": [_command_view(command) for command in plan.commands],
                "unresolved_spans": list(plan.unresolved_spans),
                "fully_parsed": plan.fully_parsed,
                "outcome": llm_outcome,
                "discarded_intent": discarded_intent,
                "llm": _command_view(llm_raw) if llm_raw is not None else None,
                "final": _command_view(final_command) if final_command is not None else None,
                "merge_changed_intent": bool(
                    llm_raw is not None
                    and final_command is not None
                    and llm_raw.intent != final_command.intent
                ),
                "merge_notes": list(merge_notes),
                "resolver": list(self._resolve_log),
                "resolver_status": (result.payload or {}).get("status"),
                "resolver_reason": (result.payload or {}).get("reason"),
                "deterministic_s": deterministic_s,
                "fallback_s": fallback_s,
                "total_s": time.perf_counter() - started,
            }
        )
        llm_debug = None
        if llm_raw is not None or llm_outcome is not None:
            llm_debug = {"decision": llm_outcome or "command"}
            if llm_raw is not None:
                llm_debug.update(_command_view(llm_raw))
            if discarded_intent:
                llm_debug["discarded_intent"] = discarded_intent
        request = RequestResult(
            request_id=request_id,
            text=text,
            status=overall_status(records),
            commands=tuple(records),
            parse_reason=reason,
            fully_parsed=plan.fully_parsed,
            unresolved=plan.unresolved_spans,
            llm=llm_debug,
            merge_notes=merge_notes if merge_ran else None,
        )
        emit_request(request)
        self.last_request = request
        return result

    def warmup(self) -> None:
        """Load the model. The warmup phrase is not a benchmark request."""

        if parse_deterministic("прогрев модели", self._registry).handled:
            return
        self._nlu.parse("прогрев модели", RequestContext())


def _as_outcome(parsed: SemanticNLUOutcome | SemanticCommand) -> SemanticNLUOutcome:
    if isinstance(parsed, SemanticCommand):
        return CommandOutcome(parsed)
    return parsed


def _stopped(kind: str) -> Result:
    return Result(
        ok=False,
        command=Command(device_id="", action=""),
        error=kind,
        payload={"outcome": kind, "status": kind},
    )


def _command_view(command: SemanticCommand) -> dict:
    target = command.target
    return {
        "intent": command.intent,
        "device_type": target.device_type,
        "mention": target.mention,
        "owner": target.owner,
        "area": target.area,
        "ordinal": target.ordinal,
        "content": command.arguments.get("content"),
        "value": command.arguments.get("value"),
    }


def _execute_plan(
    pipeline: SemanticPipeline,
    plan,
    sink,
    context,
    state,
    text: str,
) -> tuple[Result, list]:
    result: Result | None = None
    records = []
    for command in plan.commands:
        target = command.target
        slots = ExplicitSlots(
            owner=target.owner,
            raw_owner=target.raw_owner,
            area=target.area,
            raw_area=target.raw_area,
            ordinal=target.ordinal,
            device_type=target.device_type,
            raw_device_type=target.raw_device_type,
        )
        resolved = pipeline._resolver.resolve(
            command,
            context=context,
            state=state,
            text=text,
            slots=slots,
            nlu_intent=command.intent,
            normalization=plan.evidence,
        )
        pipeline._resolve_log.append(f"{resolved.status}/{resolved.reason or ''}")
        result = pipeline._executor.execute(resolved, sink)
        records.append(_recorded(pipeline, "deterministic", "command", command, resolved, result))
    if result is None:
        raise RuntimeError("fully parsed plan has no command")
    return result, records


def _recorded(pipeline: SemanticPipeline, source: str, decision: str, semantic, resolved, result):
    device = None
    if resolved.execution_target_id:
        device = pipeline._registry.get(resolved.execution_target_id)
    return command_from_result(
        source=source,
        decision=decision,
        result=result,
        semantic=semantic,
        resolution_status=resolved.status,
        device_id=resolved.execution_target_id,
        resolved_type=device.type if device is not None else None,
        resolved_area=device.area if device is not None else None,
        candidates=tuple(resolved.candidates),
        reason=resolved.reason,
    )


def _merge(
    text: str,
    deterministic: DeterministicParse,
    parsed: SemanticCommand,
    registry: DeviceRegistry,
) -> tuple[SemanticCommand, ExplicitSlots, tuple[str, ...]]:
    command, slots, notes = apply_explicit(text, parsed, registry, authoritative_intent=True)
    if not deterministic.value or command.arguments.get("value") == deterministic.value:
        return command, slots, notes
    arguments = dict(command.arguments)
    arguments["value"] = deterministic.value
    return (
        SemanticCommand(intent=command.intent, target=command.target, arguments=arguments),
        slots,
        notes + ("value from text",),
    )
