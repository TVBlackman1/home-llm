from __future__ import annotations

import re
from typing import Mapping

from ollama_runner.inventory.registry import (
    DeviceRegistry,
    RegistryDevice,
    canonicalize_device_type,
    relation_supports,
)
from ollama_runner.resolve.active import SESSION_INTENTS, ActiveSession
from ollama_runner.semantic import (
    DeviceRuntime,
    RequestContext,
    ResolutionTrace,
    ResolvedCommand,
    ScoreWeights,
    SemanticCommand,
    Target,
)


_ACTIVE_WHEN_PLAYING = frozenset({
    "media.pause",
    "media.next",
    "media.previous",
    "media.stop",
    "volume.increase",
    "volume.decrease",
    "volume.set",
})

# Used only when the device cannot do the asked power action itself.
_POWER_FALLBACK = {
    "device.turn_on": "media.play",
    "device.turn_off": "media.stop",
}


def _phrase_key(text: str) -> str:
    tokens = re.findall(r"[0-9a-zа-яе]+", text.casefold().replace("ё", "е"))
    return " ".join(sorted(tokens))


class CapabilityResolver:
    def __init__(
        self,
        registry: DeviceRegistry,
        weights: ScoreWeights | None = None,
    ) -> None:
        self._registry = registry
        self._weights = weights or ScoreWeights()

    def resolve(
        self,
        command: SemanticCommand,
        *,
        context: RequestContext | None = None,
        state: Mapping[str, DeviceRuntime] | None = None,
        text: str = "",
    ) -> ResolvedCommand:
        context = context or RequestContext()
        state = dict(state or {})
        weights = self._weights
        target = self._canonicalize_target(command.target)
        command = SemanticCommand(intent=command.intent, target=target, arguments=command.arguments)

        owner_id = self._registry.owner_id_for(target.owner) if target.owner else None
        owner_unknown = bool(target.owner) and owner_id is None
        constraints = self._constraint_lines(command, owner_unknown)

        if owner_unknown:
            return self._finish(
                command,
                text=text,
                status="unresolved",
                constraints=constraints,
                candidates=(),
                capability_lines=("owner is not in the registry",),
                state_lines=self._state_lines(state),
                score_lines=(),
                semantic_id=None,
                execution_id=None,
                tied=(),
            )

        pool = [
            device
            for device in self._registry.devices()
            if self._matches_explicit(device, target, owner_id)
        ]
        alias_ids = self._alias_hits(pool, target.mention)
        if alias_ids:
            pool = [device for device in pool if device.id in alias_ids]

        supported: list[tuple[RegistryDevice, RegistryDevice, str, str]] = []
        capability_lines: list[str] = []
        show_rejections = len(pool) <= 8
        for device in pool:
            execution, via, effective = self._execution_device(device, command.intent)
            if execution is None:
                if show_rejections:
                    capability_lines.append(f"{device.id} lacks {command.intent}")
                continue
            if effective != command.intent:
                capability_lines.append(
                    f"{device.id} lacks {command.intent}, uses {effective}"
                )
            elif via:
                capability_lines.append(
                    f"{device.id} lacks {command.intent}, {via} supports {command.intent}"
                )
            else:
                capability_lines.append(f"{device.id} supports {command.intent}")
            supported.append((device, execution, via, effective))

        state_lines = self._state_lines(state)
        session = self._session_decision(
            command,
            supported,
            state,
            text=text,
            constraints=constraints,
            capability_lines=tuple(capability_lines),
            state_lines=state_lines,
        )
        if session is not None:
            return session

        scored: list[tuple[int, RegistryDevice, RegistryDevice, str, tuple[str, ...]]] = []
        for device, execution, via, effective in supported:
            score, reasons = self._score(
                device,
                command,
                context,
                state,
                owner_id,
                alias_hit=device.id in alias_ids,
                via=via,
            )
            scored.append((score, device, execution, effective, reasons))

        scored.sort(key=lambda item: (-item[0], item[1].id))
        score_lines = tuple(
            f"{device.id} {' '.join(reasons) if reasons else '+0'}"
            for _, device, _, _, reasons in scored
        )

        supported_ids = tuple(device.id for _, device, _, _, _ in scored)
        if not scored:
            return self._finish(
                command,
                text=text,
                status="unresolved",
                constraints=constraints,
                candidates=tuple(device.id for device in pool),
                capability_lines=tuple(capability_lines) or ("no device supports the capability",),
                state_lines=state_lines,
                score_lines=score_lines,
                semantic_id=None,
                execution_id=None,
                tied=(),
            )

        best = scored[0][0]
        tied = [item for item in scored if best - item[0] < weights.ambiguity_margin]
        if len(tied) > 1:
            return self._finish(
                command,
                text=text,
                status="ambiguous",
                constraints=constraints,
                candidates=supported_ids,
                capability_lines=tuple(capability_lines),
                state_lines=state_lines,
                score_lines=score_lines,
                semantic_id=None,
                execution_id=None,
                tied=tuple(item[1].id for item in tied),
            )

        _, semantic, execution, effective, _ = scored[0]
        return self._finish(
            command,
            text=text,
            status="resolved",
            constraints=constraints,
            candidates=supported_ids,
            capability_lines=tuple(capability_lines),
            state_lines=state_lines,
            score_lines=score_lines,
            semantic_id=semantic.id,
            execution_id=execution.id,
            tied=(semantic.id,),
            intent=effective,
        )

    def _canonicalize_target(self, target: Target) -> Target:
        device_type = canonicalize_device_type(target.device_type)
        area = target.area.strip() if target.area else None
        owner = target.owner.strip() if target.owner else None
        mention = target.mention.strip() if target.mention else None
        explicit = any((device_type, mention, owner, area, target.ordinal is not None))
        return Target(
            device_type=device_type,
            mention=mention,
            owner=owner,
            area=area.casefold() if area else None,
            ordinal=target.ordinal,
            explicit=explicit,
        )

    def _matches_explicit(
        self,
        device: RegistryDevice,
        target: Target,
        owner_id: str | None,
    ) -> bool:
        if target.device_type and device.type != target.device_type:
            return False
        if target.owner:
            if device.owner_id != owner_id:
                return False
        if target.area and device.area.casefold() != target.area:
            return False
        if target.ordinal is not None and device.ordinal != target.ordinal:
            return False
        return True

    def _alias_hits(self, devices: list[RegistryDevice], mention: str | None) -> set[str]:
        if not mention:
            return set()
        key = _phrase_key(mention)
        if not key:
            return set()
        return {
            device.id
            for device in devices
            if any(_phrase_key(alias) == key for alias in device.aliases)
        }

    def _execution_device(
        self,
        device: RegistryDevice,
        intent: str,
    ) -> tuple[RegistryDevice | None, str, str]:
        if intent in device.capabilities:
            return device, "", intent
        fallback = _POWER_FALLBACK.get(intent)
        if fallback and fallback in device.capabilities:
            return device, "", fallback
        for relation_type, target_id in device.relationships:
            if not relation_supports(relation_type, intent):
                continue
            other = self._registry.get(target_id)
            if other is not None and intent in other.capabilities:
                return other, f"{relation_type} → {other.id}", intent
        return None, "", intent

    def _session_decision(
        self,
        command: SemanticCommand,
        supported: list[tuple[RegistryDevice, RegistryDevice, str, str]],
        state: Mapping[str, DeviceRuntime],
        *,
        text: str,
        constraints: tuple[str, ...],
        capability_lines: tuple[str, ...],
        state_lines: tuple[str, ...],
    ) -> ResolvedCommand | None:
        if command.intent not in SESSION_INTENTS or len(supported) == 1:
            return None

        session = ActiveSession(state)
        active = [
            item
            for item in supported
            if session.is_active(item[0].id, command.intent)
        ]
        if len(active) == 1:
            device, execution, _, effective = active[0]
            note = "currently paused" if command.intent == "media.resume" else "currently playing"
            return self._finish(
                command,
                text=text,
                status="resolved",
                constraints=constraints,
                candidates=(device.id,),
                capability_lines=capability_lines,
                state_lines=state_lines,
                score_lines=(f"{device.id} {note}",),
                semantic_id=device.id,
                execution_id=execution.id,
                tied=(device.id,),
                intent=effective,
                reason="active_device",
            )

        reason = "no_active_device" if not active else "multiple_active_devices"
        active_ids = tuple(item[0].id for item in active)
        return self._finish(
            command,
            text=text,
            status="clarify",
            constraints=constraints,
            candidates=active_ids,
            capability_lines=capability_lines,
            state_lines=state_lines,
            score_lines=(reason,),
            semantic_id=None,
            execution_id=None,
            tied=active_ids,
            reason=reason,
        )

    def _score(
        self,
        device: RegistryDevice,
        command: SemanticCommand,
        context: RequestContext,
        state: Mapping[str, DeviceRuntime],
        owner_id: str | None,
        *,
        alias_hit: bool,
        via: str,
    ) -> tuple[int, tuple[str, ...]]:
        weights = self._weights
        target = command.target
        score = 0
        reasons: list[str] = []

        def add(points: int, reason: str) -> None:
            nonlocal score
            if points:
                score += points
                reasons.append(f"+{points} {reason}")

        if target.device_type and device.type == target.device_type:
            add(weights.explicit_device_type, "explicit device type")
        if target.owner and device.owner_id == owner_id:
            add(weights.explicit_owner, "explicit owner")
        if target.area and device.area.casefold() == target.area:
            add(weights.explicit_area, "explicit area")
        if target.ordinal is not None and device.ordinal == target.ordinal:
            add(weights.explicit_ordinal, "explicit ordinal")
        if alias_hit:
            add(weights.exact_alias, "exact alias")
        if via:
            add(weights.relationship_match, f"relationship {via}")

        runtime = state.get(device.id)
        media = runtime.media if runtime else None
        if command.intent in _ACTIVE_WHEN_PLAYING and media == "playing":
            add(weights.currently_active, "currently playing")
        elif command.intent == "media.resume" and media == "paused":
            add(weights.currently_active, "currently paused")

        if not target.area and context.source_area and device.area.casefold() == context.source_area.casefold():
            add(weights.request_origin_area, "request area")
        if not target.area and not device.area:
            add(weights.unscoped_area, "unscoped area")
        if not target.owner and device.owner_id in {"", "common"}:
            add(weights.unscoped_owner, "unscoped owner")

        context_owner = self._registry.owner_id_for(context.user)
        if not target.owner and context_owner and device.owner_id == context_owner:
            add(weights.user_default, "user default")

        return score, tuple(reasons)

    def _constraint_lines(self, command: SemanticCommand, owner_unknown: bool) -> tuple[str, ...]:
        target = command.target
        lines = [f"capability {command.intent} REQUIRED"]
        if target.device_type:
            lines.append(f"device_type={target.device_type}")
        if target.owner:
            lines.append(f"owner={target.owner}")
        if owner_unknown:
            lines.append("owner is unknown")
        if target.area:
            lines.append(f"area={target.area}")
        if target.ordinal is not None:
            lines.append(f"ordinal={target.ordinal}")
        if target.mention:
            lines.append(f"mention={target.mention}")
        if target.unconstrained():
            lines.append("target unspecified")
        return tuple(lines)

    def _state_lines(self, state: Mapping[str, DeviceRuntime]) -> tuple[str, ...]:
        if not state:
            return ()
        return tuple(
            f"{device_id} media={runtime.media or '—'}"
            for device_id, runtime in sorted(state.items())
        )

    def _finish(
        self,
        command: SemanticCommand,
        *,
        text: str,
        status: str,
        constraints: tuple[str, ...],
        candidates: tuple[str, ...],
        capability_lines: tuple[str, ...],
        state_lines: tuple[str, ...],
        score_lines: tuple[str, ...],
        semantic_id: str | None,
        execution_id: str | None,
        tied: tuple[str, ...],
        intent: str | None = None,
        reason: str = "",
    ) -> ResolvedCommand:
        trace = ResolutionTrace(
            text=text,
            intent=intent or command.intent,
            target=command.target,
            constraint_lines=constraints,
            candidate_ids=candidates,
            capability_lines=capability_lines,
            state_lines=state_lines,
            score_lines=score_lines,
            status=status,
            semantic_target=semantic_id,
            execution_target=execution_id,
            reason=reason,
        )
        owner = ""
        area = ""
        if command.target.owner:
            owner = self._registry.owner_name(self._registry.owner_id_for(command.target.owner))
        if command.target.area:
            area = command.target.area
        reported = tied if status in {"ambiguous", "clarify"} else candidates
        return ResolvedCommand(
            status=status,
            intent=intent or command.intent,
            reason=reason,
            semantic_target_id=semantic_id,
            execution_target_id=execution_id,
            arguments=dict(command.arguments),
            candidates=reported,
            owner=owner,
            area=area,
            trace=trace,
        )
