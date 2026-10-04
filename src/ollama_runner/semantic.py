from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
NOT_FOUND = "not_found"
UNSUPPORTED = "unsupported"
CLARIFY = "clarify"


@dataclass(frozen=True)
class ExplicitSlots:
    """Slots read from the utterance itself, before the model is trusted."""

    owner: str | None = None
    raw_owner: str | None = None
    area: str | None = None
    raw_area: str | None = None
    ordinal: int | None = None
    device_type: str | None = None
    raw_device_type: str | None = None


@dataclass(frozen=True)
class Target:
    device_type: str | None = None
    mention: str | None = None
    owner: str | None = None
    area: str | None = None
    ordinal: int | None = None
    explicit: bool = False
    raw_owner: str | None = None
    raw_area: str | None = None
    raw_device_type: str | None = None

    def unconstrained(self) -> bool:
        return not any((
            self.device_type,
            self.mention,
            self.owner,
            self.area,
            self.ordinal is not None,
            self.explicit,
        ))


@dataclass(frozen=True)
class SemanticCommand:
    intent: str
    target: Target = field(default_factory=Target)
    arguments: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandOutcome:
    """Fallback decided the utterance is a self-contained household command."""

    command: SemanticCommand
    kind: str = "command"


@dataclass(frozen=True)
class NotCommandOutcome:
    """Fallback decided the utterance is not a smart-home command.

    A discarded intent is instrumentation only. It is not a command.
    """

    kind: str = "not_command"
    discarded_intent: str | None = None


@dataclass(frozen=True)
class NeedsContextOutcome:
    """Fallback decided the utterance may be a continuation, but the action is not in the text."""

    kind: str = "needs_context"
    discarded_intent: str | None = None


SemanticNLUOutcome = CommandOutcome | NotCommandOutcome | NeedsContextOutcome


@dataclass(frozen=True)
class SemanticPlan:
    """One utterance expanded into atomic commands before the resolver.

    A coordinated area or device is already a separate command here.
    `fully_parsed` is false when any span was left unresolved: the caller
    must not execute the understood prefix on its own.
    """

    commands: tuple[SemanticCommand, ...] = ()
    fully_parsed: bool = False
    unresolved_spans: tuple[str, ...] = ()
    reason: str = "no_intent_cue"
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestContext:
    source_device: str | None = None
    source_area: str | None = None
    user: str | None = None


@dataclass(frozen=True)
class DeviceRuntime:
    media: str | None = None
    power: str | None = None


@dataclass(frozen=True)
class ScoreWeights:
    """Explicit fields are hard filters. These weights explain the survivors
    and rank them. Contextual weights are what actually separate candidates.
    """

    exact_alias: int = 100
    explicit_device_type: int = 80
    explicit_owner: int = 50
    explicit_area: int = 50
    explicit_ordinal: int = 50
    currently_active: int = 30
    request_origin_area: int = 20
    relationship_match: int = 20
    user_default: int = 10
    unscoped_area: int = 5
    unscoped_owner: int = 5
    ambiguity_margin: int = 1


@dataclass(frozen=True)
class ResolutionTrace:
    text: str
    intent: str
    target: Target
    constraint_lines: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    capability_lines: tuple[str, ...]
    state_lines: tuple[str, ...]
    score_lines: tuple[str, ...]
    status: str
    semantic_target: str | None
    execution_target: str | None
    reason: str = ""
    skill: str | None = None
    ha_action: str | None = None
    nlu_intent: str = ""
    explicit_lines: tuple[str, ...] = ()
    normalization_lines: tuple[str, ...] = ()
    policy: str = ""

    def with_execution(self, *, skill: str, ha_action: str) -> ResolutionTrace:
        return replace(self, skill=skill, ha_action=ha_action)


@dataclass(frozen=True)
class ResolvedCommand:
    status: str
    intent: str
    semantic_target_id: str | None
    execution_target_id: str | None
    arguments: Mapping[str, str]
    candidates: tuple[str, ...]
    owner: str = ""
    area: str = ""
    reason: str = ""
    missing: tuple[str, ...] = ()
    trace: ResolutionTrace | None = None


def format_target(target: Target) -> str:
    if target.unconstrained():
        return "unspecified"

    parts = [
        f"device_type={target.device_type or ''}",
        f"mention={target.mention or ''}",
        f"owner={target.owner or ''}",
        f"area={target.area or ''}",
        f"ordinal={target.ordinal if target.ordinal is not None else ''}",
        f"explicit={str(target.explicit).lower()}",
    ]
    return " ".join(parts)


def format_trace(trace: ResolutionTrace) -> str:
    lines = [
        "INPUT:",
        trace.text or "—",
        "EXPLICIT SLOTS:",
        *(trace.explicit_lines or ("—",)),
        "NLU:",
        f"intent={trace.nlu_intent or trace.intent}",
        f"target={format_target(trace.target)}",
        "NORMALIZATION:",
        *(trace.normalization_lines or ("—",)),
        "POLICY:",
        trace.policy or "—",
        "CONSTRAINTS:",
        *(trace.constraint_lines or ("—",)),
        "CANDIDATES:",
        *(trace.candidate_ids or ("—",)),
        "CAPABILITY FILTER:",
        *(trace.capability_lines or ("—",)),
        "STATE:",
        *(trace.state_lines or ("—",)),
        "SCORING:",
        *(trace.score_lines or ("—",)),
        "STATUS:",
        trace.status,
        "REASON:",
        trace.reason or "—",
        "SEMANTIC TARGET:",
        trace.semantic_target or "—",
        "EXECUTION TARGET:",
        trace.execution_target or "—",
    ]
    if trace.skill:
        lines.extend(("SKILL:", trace.skill))
    if trace.ha_action:
        lines.extend(("HA ACTION:", trace.ha_action))
    return "\n".join(lines)
