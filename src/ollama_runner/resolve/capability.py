from __future__ import annotations

import re
from typing import Mapping

from ollama_runner.inventory.registry import (
    DeviceRegistry,
    RegistryDevice,
    canonicalize_area,
    canonicalize_device_type,
    device_type_from_text,
    relation_supports,
    stem_token,
)
from ollama_runner.resolve.active import ActiveSession, policy_for
from ollama_runner.semantic import (
    AMBIGUOUS,
    CLARIFY,
    NOT_FOUND,
    RESOLVED,
    UNSUPPORTED,
    DeviceRuntime,
    ExplicitSlots,
    RequestContext,
    ResolutionTrace,
    ResolvedCommand,
    ScoreWeights,
    SemanticCommand,
    Target,
)


# Used only when the device cannot do the asked power action itself.
_POWER_FALLBACK = {
    "device.turn_on": "media.play",
    "device.turn_off": "media.stop",
}
_FANOUT = frozenset({
    "brightness.set",
    "brightness.increase",
    "brightness.decrease",
})


def _phrase_key(text: str) -> str:
    tokens = re.findall(r"[0-9a-zа-яе]+", text.casefold().replace("ё", "е"))
    return " ".join(sorted(stem_token(token) for token in tokens))


class CapabilityResolver:
    def __init__(
        self,
        registry: DeviceRegistry,
        weights: ScoreWeights | None = None,
        *,
        members: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._registry = registry
        self._weights = weights or ScoreWeights()
        self._members = {
            entity_id: tuple(contained)
            for entity_id, contained in (members or {}).items()
            if contained
        }
        self._slots = ExplicitSlots()
        self._nlu_intent = ""
        self._normalization: tuple[str, ...] = ()
        self._policy = ""

    def resolve(
        self,
        command: SemanticCommand,
        *,
        context: RequestContext | None = None,
        state: Mapping[str, DeviceRuntime] | None = None,
        text: str = "",
        slots: ExplicitSlots | None = None,
        nlu_intent: str = "",
        normalization: tuple[str, ...] = (),
    ) -> ResolvedCommand:
        context = context or RequestContext()
        state = dict(state or {})
        weights = self._weights
        target = self._canonicalize_target(command.target)
        command = SemanticCommand(intent=command.intent, target=target, arguments=command.arguments)
        self._slots = slots or ExplicitSlots()
        self._nlu_intent = nlu_intent or command.intent
        self._normalization = normalization

        owner_id = self._registry.owner_id_for(target.owner) if target.owner else None
        owner_unknown = bool(target.owner) and owner_id is None
        constraints = self._constraint_lines(command, owner_unknown)

        if command.intent == "content.play":
            return self._finish(
                command,
                text=text,
                status=AMBIGUOUS,
                constraints=constraints,
                candidates=(),
                capability_lines=("content.play does not distinguish audio and video",),
                state_lines=self._state_lines(state),
                score_lines=(),
                semantic_id=None,
                execution_id=None,
                tied=(),
                reason="media_type",
                missing=("media_type",),
            )

        if owner_unknown:
            return self._finish(
                command,
                text=text,
                status=NOT_FOUND,
                constraints=constraints,
                candidates=(),
                capability_lines=("owner is not in the registry",),
                state_lines=self._state_lines(state),
                score_lines=(),
                semantic_id=None,
                execution_id=None,
                tied=(),
                reason="unknown_owner",
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
        if not pool:
            return self._finish(
                command,
                text=text,
                status=NOT_FOUND,
                constraints=constraints,
                candidates=(),
                capability_lines=tuple(capability_lines) or ("no device matches explicit constraints",),
                state_lines=state_lines,
                score_lines=(),
                semantic_id=None,
                execution_id=None,
                tied=(),
                reason="no_match",
            )
        if not supported:
            named = bool(command.target.device_type or command.target.mention)
            return self._finish(
                command,
                text=text,
                status=UNSUPPORTED if named else NOT_FOUND,
                constraints=constraints,
                candidates=tuple(device.id for device in pool),
                capability_lines=tuple(capability_lines) or ("no device supports the capability",),
                state_lines=state_lines,
                score_lines=(),
                semantic_id=None,
                execution_id=None,
                tied=(),
                reason="capability" if named else "no_capable_device",
                missing=(command.intent,) if named else (),
            )

        if command.intent == "device.toggle":
            return self._light_toggle(
                command,
                supported,
                state,
                text=text,
                constraints=constraints,
                capability_lines=tuple(capability_lines),
                state_lines=state_lines,
            )
        if command.intent == "media.toggle":
            return self._media_toggle(
                command,
                supported,
                state,
                text=text,
                constraints=constraints,
                capability_lines=tuple(capability_lines),
                state_lines=state_lines,
            )

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

        fanout = self._fanout(
            command,
            supported,
            alias_ids,
            text=text,
            constraints=constraints,
            capability_lines=tuple(capability_lines),
            state_lines=state_lines,
        )
        if fanout is not None:
            return fanout

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
        best = scored[0][0]
        tied = [item for item in scored if best - item[0] < weights.ambiguity_margin]
        if len(tied) > 1:
            tied_devices = tuple(item[1] for item in tied)
            return self._finish(
                command,
                text=text,
                status=AMBIGUOUS,
                constraints=constraints,
                candidates=supported_ids,
                capability_lines=tuple(capability_lines),
                state_lines=state_lines,
                score_lines=score_lines,
                semantic_id=None,
                execution_id=None,
                tied=tuple(device.id for device in tied_devices),
                missing=self._missing(tied_devices, command.target),
            )

        _, semantic, execution, effective, _ = scored[0]
        return self._finish(
            command,
            text=text,
            status=RESOLVED,
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
        device_type = canonicalize_device_type(target.device_type) or target.device_type
        if target.device_type and canonicalize_device_type(target.device_type):
            device_type = canonicalize_device_type(target.device_type)
        area = canonicalize_area(target.area) if target.area else None
        if target.area and area is None:
            area = target.area.strip().casefold()
        owner = self._registry.canonicalize_owner(target.owner) if target.owner else None
        if target.owner and owner is None:
            owner = target.owner.strip()
        mention = target.mention.strip() if target.mention else None
        explicit = any((device_type, mention, owner, area, target.ordinal is not None))
        return Target(
            device_type=device_type,
            mention=mention,
            owner=owner,
            area=area,
            ordinal=target.ordinal,
            explicit=explicit,
            raw_owner=target.raw_owner,
            raw_area=target.raw_area,
            raw_device_type=target.raw_device_type,
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
        if intent == "device.toggle":
            if "device.turn_on" in device.capabilities and "device.turn_off" in device.capabilities:
                return device, "", intent
            return None, "", intent
        if intent == "media.toggle":
            if "media.pause" in device.capabilities or "media.resume" in device.capabilities:
                return device, "", intent
            return None, "", intent
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
        policy = policy_for(command.intent)
        if policy is None or len(supported) == 1:
            self._policy = ""
            return None
        self._policy = policy.description

        session = ActiveSession(state)
        active = [
            item
            for item in supported
            if session.is_active(item[0].id, item[1].id, policy)
        ]
        if len(active) == 1:
            device, execution, _, effective = active[0]
            note = "currently paused" if "paused" in policy.active_media and "playing" not in policy.active_media else "currently playing"
            return self._finish(
                command,
                text=text,
                status=RESOLVED,
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

        if not active:
            return self._finish(
                command,
                text=text,
                status=CLARIFY,
                constraints=constraints,
                candidates=(),
                capability_lines=capability_lines,
                state_lines=state_lines,
                score_lines=("no_active_device",),
                semantic_id=None,
                execution_id=None,
                tied=(),
                reason="no_active_device",
                missing=("active_device",),
            )

        active_devices = tuple(item[0] for item in active)
        return self._finish(
            command,
            text=text,
            status=AMBIGUOUS,
            constraints=constraints,
            candidates=tuple(device.id for device in active_devices),
            capability_lines=capability_lines,
            state_lines=state_lines,
            score_lines=("multiple_active_devices",),
            semantic_id=None,
            execution_id=None,
            tied=tuple(device.id for device in active_devices),
            reason="multiple_active_devices",
            missing=self._missing(active_devices, command.target) or ("device",),
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
        missing: tuple[str, ...] = (),
        execution_target_ids: tuple[str, ...] = (),
        execution_intents: tuple[str, ...] = (),
    ) -> ResolvedCommand:
        slots = self._slots
        explicit_lines = (
            f"owner={slots.owner or 'null'}",
            f"area={slots.area or 'null'}",
            f"ordinal={slots.ordinal if slots.ordinal is not None else 'null'}",
            f"device_type={slots.device_type or 'null'}",
        )
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
            nlu_intent=self._nlu_intent,
            explicit_lines=explicit_lines,
            normalization_lines=self._normalization,
            policy=self._policy,
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
            missing=missing,
            trace=trace,
            execution_target_ids=execution_target_ids,
            execution_intents=execution_intents,
        )

    def _concrete(self, target: Target, alias_ids: set[str]) -> bool:
        """An ordinal or a name that is not just the device type picks one device."""

        if target.ordinal is not None:
            return True
        if not target.mention or not alias_ids:
            return False
        found = device_type_from_text(target.mention)
        if found is not None and _phrase_key(found[1]) == _phrase_key(target.mention):
            return False
        return True

    def _covered(self, selected: set[str]) -> set[str]:
        """Candidates transitively contained by another selected candidate."""

        covered: set[str] = set()
        for group_id in selected:
            stack = list(self._members.get(group_id, ()))
            seen: set[str] = set()
            while stack:
                member = stack.pop()
                if member in seen:
                    continue
                seen.add(member)
                if member in selected:
                    covered.add(member)
                stack.extend(self._members.get(member, ()))
        return covered

    def _maximal(
        self,
        supported: list[tuple[RegistryDevice, RegistryDevice, str, str]],
    ) -> list[tuple[RegistryDevice, RegistryDevice, str, str]]:
        selected = {item[0].id for item in supported}
        covered = self._covered(selected)
        kept = [item for item in supported if item[0].id not in covered]
        kept.sort(key=lambda item: item[0].id)
        return kept

    def _fanout(
        self,
        command: SemanticCommand,
        supported: list[tuple[RegistryDevice, RegistryDevice, str, str]],
        alias_ids: set[str],
        *,
        text: str,
        constraints: tuple[str, ...],
        capability_lines: tuple[str, ...],
        state_lines: tuple[str, ...],
    ) -> ResolvedCommand | None:
        area_power = (
            command.intent in {"device.turn_on", "device.turn_off"}
            and command.target.device_type == "light"
            and bool(command.target.area)
        )
        if command.intent not in _FANOUT and not area_power:
            return None
        if self._concrete(command.target, alias_ids):
            return None
        maximal = self._maximal(supported)
        types = {item[0].type for item in maximal}
        if len(types) > 1:
            devices = tuple(item[0] for item in maximal)
            return self._finish(
                command,
                text=text,
                status=AMBIGUOUS,
                constraints=constraints,
                candidates=tuple(device.id for device in devices),
                capability_lines=capability_lines,
                state_lines=state_lines,
                score_lines=("multiple device types",),
                semantic_id=None,
                execution_id=None,
                tied=tuple(device.id for device in devices),
                reason="device_type",
                missing=("device_type",),
            )
        if len(maximal) == 1:
            device, execution, _, effective = maximal[0]
            return self._finish(
                command,
                text=text,
                status=RESOLVED,
                constraints=constraints,
                candidates=(device.id,),
                capability_lines=capability_lines,
                state_lines=state_lines,
                score_lines=(f"{device.id} maximal target",),
                semantic_id=device.id,
                execution_id=execution.id,
                tied=(device.id,),
                intent=effective,
            )
        ids = tuple(item[0].id for item in maximal)
        return self._finish(
            command,
            text=text,
            status=RESOLVED,
            constraints=constraints,
            candidates=ids,
            capability_lines=capability_lines,
            state_lines=state_lines,
            score_lines=tuple(f"{device_id} maximal target" for device_id in ids),
            semantic_id=ids[0],
            execution_id=ids[0],
            tied=ids,
            reason="maximal_targets",
            execution_target_ids=ids,
            execution_intents=tuple(command.intent for _ in ids),
        )

    def _light_toggle(
        self,
        command: SemanticCommand,
        supported: list[tuple[RegistryDevice, RegistryDevice, str, str]],
        state: Mapping[str, DeviceRuntime],
        *,
        text: str,
        constraints: tuple[str, ...],
        capability_lines: tuple[str, ...],
        state_lines: tuple[str, ...],
    ) -> ResolvedCommand:
        maximal = self._maximal(supported)
        if not maximal:
            return self._finish(
                command,
                text=text,
                status=NOT_FOUND,
                constraints=constraints,
                candidates=(),
                capability_lines=capability_lines,
                state_lines=state_lines,
                score_lines=(),
                semantic_id=None,
                execution_id=None,
                tied=(),
                reason="no_capable_device",
            )
        ids: list[str] = []
        intents: list[str] = []
        notes: list[str] = []
        for device, _, _, _ in maximal:
            runtime = state.get(device.id)
            power = runtime.power if runtime is not None else None
            if power == "on":
                intent = "device.turn_off"
            elif power == "off":
                intent = "device.turn_on"
            else:
                intent = "device.toggle"
            ids.append(device.id)
            intents.append(intent)
            notes.append(f"{device.id} power={power or 'unknown'} → {intent}")
        return self._finish(
            command,
            text=text,
            status=RESOLVED,
            constraints=constraints,
            candidates=tuple(ids),
            capability_lines=capability_lines,
            state_lines=state_lines,
            score_lines=tuple(notes),
            semantic_id=ids[0],
            execution_id=ids[0],
            tied=tuple(ids),
            intent=intents[0] if len(ids) == 1 else command.intent,
            reason="maximal_targets",
            execution_target_ids=tuple(ids),
            execution_intents=tuple(intents),
        )

    def _media_toggle(
        self,
        command: SemanticCommand,
        supported: list[tuple[RegistryDevice, RegistryDevice, str, str]],
        state: Mapping[str, DeviceRuntime],
        *,
        text: str,
        constraints: tuple[str, ...],
        capability_lines: tuple[str, ...],
        state_lines: tuple[str, ...],
    ) -> ResolvedCommand:
        active: list[tuple[RegistryDevice, RegistryDevice, str]] = []
        for device, execution, _, _ in supported:
            runtime = state.get(device.id) or state.get(execution.id)
            media = runtime.media if runtime is not None else None
            if media == "playing" and "media.pause" in device.capabilities:
                active.append((device, execution, "media.pause"))
            elif media == "paused" and "media.resume" in device.capabilities:
                active.append((device, execution, "media.resume"))
        if not active:
            return self._finish(
                command,
                text=text,
                status=CLARIFY,
                constraints=constraints,
                candidates=(),
                capability_lines=capability_lines,
                state_lines=state_lines,
                score_lines=("no_active_device",),
                semantic_id=None,
                execution_id=None,
                tied=(),
                reason="no_active_device",
                missing=("active_device",),
            )
        if len(active) > 1:
            devices = tuple(item[0] for item in active)
            return self._finish(
                command,
                text=text,
                status=AMBIGUOUS,
                constraints=constraints,
                candidates=tuple(device.id for device in devices),
                capability_lines=capability_lines,
                state_lines=state_lines,
                score_lines=("multiple_active_devices",),
                semantic_id=None,
                execution_id=None,
                tied=tuple(device.id for device in devices),
                reason="multiple_active_devices",
                missing=("device",),
            )
        device, execution, intent = active[0]
        note = "currently playing" if intent == "media.pause" else "currently paused"
        return self._finish(
            command,
            text=text,
            status=RESOLVED,
            constraints=constraints,
            candidates=(device.id,),
            capability_lines=capability_lines,
            state_lines=state_lines,
            score_lines=(f"{device.id} {note}",),
            semantic_id=device.id,
            execution_id=execution.id,
            tied=(device.id,),
            intent=intent,
            reason="active_device",
        )

    def _missing(self, devices: tuple[RegistryDevice, ...], target: Target) -> tuple[str, ...]:
        missing: list[str] = []
        if not target.area and len({device.area for device in devices}) > 1:
            missing.append("area")
        if not target.owner and len({device.owner_id for device in devices}) > 1:
            missing.append("owner")
        if target.ordinal is None and len({device.ordinal for device in devices}) > 1:
            missing.append("ordinal")
        if not target.device_type and len({device.type for device in devices}) > 1:
            missing.append("device_type")
        return tuple(missing)
