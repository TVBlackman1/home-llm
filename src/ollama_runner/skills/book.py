from __future__ import annotations

from dataclasses import dataclass

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.semantic import ResolvedCommand, format_trace
from ollama_runner.types import Command, Result


LEGACY_ACTION = {
    "device.turn_on": "on",
    "device.turn_off": "off",
    "brightness.increase": "increase",
    "brightness.decrease": "decrease",
    "brightness.set": "set_state",
    "color.set": "set_state",
    "color_temperature.set": "set_state",
    "color_temperature.warmer": "increase",
    "color_temperature.cooler": "decrease",
    "volume.increase": "increase",
    "volume.decrease": "decrease",
    "volume.set": "set_state",
    "media.play": "on",
    "media.pause": "stop",
    "media.resume": "on",
    "media.stop": "stop",
    "media.next": "increase",
    "media.previous": "decrease",
    "media.seek_forward": "increase",
    "media.seek_backward": "decrease",
    "audio.play": "run_content",
    "video.play": "run_content",
    "content.play": "run_content",
    "photos.show": "run_content",
    "app.launch": "on",
    "screen.share": "on",
    "pc.turn_on": "on",
    "pc.turn_off": "off",
}

HA_ACTION = {
    "device.turn_on": "homeassistant.turn_on",
    "device.turn_off": "homeassistant.turn_off",
    "brightness.increase": "light.brightness_increase",
    "brightness.decrease": "light.brightness_decrease",
    "brightness.set": "light.turn_on",
    "color.set": "light.turn_on",
    "color_temperature.set": "light.turn_on",
    "color_temperature.warmer": "light.turn_on",
    "color_temperature.cooler": "light.turn_on",
    "volume.increase": "media_player.volume_up",
    "volume.decrease": "media_player.volume_down",
    "volume.set": "media_player.volume_set",
    "media.play": "media_player.media_play",
    "media.pause": "media_player.media_pause",
    "media.resume": "media_player.media_play",
    "media.stop": "media_player.media_stop",
    "media.next": "media_player.media_next_track",
    "media.previous": "media_player.media_previous_track",
    "media.seek_forward": "media_player.media_seek",
    "media.seek_backward": "media_player.media_seek",
    "audio.play": "media_player.play_media",
    "video.play": "media_player.play_media",
    "content.play": "media_player.play_media",
    "photos.show": "media_player.play_media",
    "app.launch": "media_player.select_source",
    "screen.share": "media_player.play_media",
    "pc.turn_on": "wake_on_lan.send_magic_packet",
    "pc.turn_off": "homeassistant.turn_off",
}

SKILL_NAME = {
    "device.turn_on": "PowerSkill",
    "device.turn_off": "PowerSkill",
    "pc.turn_on": "PCSkill",
    "pc.turn_off": "PCSkill",
    "brightness.increase": "LightingSkill",
    "brightness.decrease": "LightingSkill",
    "brightness.set": "LightingSkill",
    "color.set": "LightingSkill",
    "color_temperature.set": "LightingSkill",
    "color_temperature.warmer": "LightingSkill",
    "color_temperature.cooler": "LightingSkill",
    "volume.increase": "VolumeSkill",
    "volume.decrease": "VolumeSkill",
    "volume.set": "VolumeSkill",
    "audio.play": "MediaSkill",
    "video.play": "MediaSkill",
    "content.play": "MediaSkill",
    "photos.show": "PhotoSkill",
    "screen.share": "ScreenShareSkill",
    "app.launch": "MediaSkill",
}


@dataclass(frozen=True)
class SkillOutcome:
    skill: str
    ha_action: str
    command: Command


_CONTENT_INTENTS = frozenset({"audio.play", "video.play", "content.play", "photos.show"})


def legacy_value(intent: str, arguments: dict) -> str:
    if intent in _CONTENT_INTENTS:
        return str(arguments.get("content") or "")
    return str(arguments.get("value") or "")


def _semantic_view(resolved: ResolvedCommand) -> dict:
    target = resolved.trace.target if resolved.trace is not None else None
    return {
        "intent": resolved.intent,
        "nlu_intent": resolved.trace.nlu_intent if resolved.trace is not None else "",
        "device_type": target.device_type if target is not None else None,
        "owner": target.owner if target is not None else None,
        "area": target.area if target is not None else None,
        "ordinal": target.ordinal if target is not None else None,
        "content": resolved.arguments.get("content", ""),
        "missing": list(resolved.missing),
    }


def skill_name(intent: str) -> str:
    if intent in SKILL_NAME:
        return SKILL_NAME[intent]
    if intent.startswith("media."):
        return "MediaSkill"
    return "UnsupportedSkill"


def render_ha_action(intent: str, execution_target: str, arguments: dict) -> str:
    service = HA_ACTION[intent]
    content = arguments.get("content")
    value = arguments.get("value")
    if content:
        return f'{service}({execution_target}, content="{content}")'
    if value:
        return f'{service}({execution_target}, value="{value}")'
    return f"{service}({execution_target})"


class SkillBook:
    def apply(self, resolved: ResolvedCommand) -> SkillOutcome:
        if resolved.execution_target_id is None:
            raise RuntimeError("skill called without an execution target")
        intent = resolved.intent
        if intent not in HA_ACTION or intent not in LEGACY_ACTION:
            raise RuntimeError(f"no HA action for {intent}")

        arguments = dict(resolved.arguments)
        name = skill_name(intent)
        ha_action = render_ha_action(intent, resolved.execution_target_id, arguments)
        command = Command(
            device_id=resolved.execution_target_id,
            action=LEGACY_ACTION[intent],
            value=legacy_value(intent, arguments),
            owner=resolved.owner,
            place=resolved.area,
        )
        return SkillOutcome(skill=name, ha_action=ha_action, command=command)


class Executor:
    """Validates a resolved command and adapts it to the existing sink."""

    def __init__(self, registry: DeviceRegistry) -> None:
        self._registry = registry
        self._skills = SkillBook()

    def execute(self, resolved: ResolvedCommand, sink) -> Result:
        trace = resolved.trace
        trace_text = format_trace(trace) if trace is not None else ""
        arguments = dict(resolved.arguments)

        if resolved.status != "resolved" or not resolved.execution_target_id:
            return Result(
                ok=False,
                command=Command(
                    device_id="",
                    action=LEGACY_ACTION.get(resolved.intent, resolved.intent),
                    value=legacy_value(resolved.intent, arguments),
                    owner=resolved.owner,
                    place=resolved.area,
                ),
                error=resolved.status,
                payload={
                    "trace": trace_text,
                    "candidates": list(resolved.candidates),
                    "missing": list(resolved.missing),
                    "status": resolved.status,
                    "reason": resolved.reason,
                    "semantic": _semantic_view(resolved),
                },
            )

        device = self._registry.get(resolved.execution_target_id)
        if device is None:
            return self._rejected(resolved, trace_text, "execution target does not exist")
        if resolved.intent not in device.capabilities:
            return self._rejected(resolved, trace_text, "execution target lacks capability")
        if resolved.intent not in HA_ACTION:
            return self._rejected(resolved, trace_text, "HA action does not exist")
        if resolved.intent in _CONTENT_INTENTS and not (arguments.get("content") or "").strip():
            return self._rejected(resolved, trace_text, "content is missing")

        outcome = self._skills.apply(resolved)
        if trace is not None:
            trace_text = format_trace(
                trace.with_execution(skill=outcome.skill, ha_action=outcome.ha_action)
            )
        result = sink.send(outcome.command)
        return Result(
            ok=result.ok,
            command=result.command,
            error=result.error,
            payload={
                "trace": trace_text,
                "skill": outcome.skill,
                "ha_action": outcome.ha_action,
                "semantic_target": resolved.semantic_target_id,
                "execution_target": resolved.execution_target_id,
                "status": resolved.status,
                "reason": resolved.reason,
                "semantic": _semantic_view(resolved),
            },
        )

    def _rejected(self, resolved: ResolvedCommand, trace_text: str, error: str) -> Result:
        return Result(
            ok=False,
            command=Command(
                device_id=resolved.execution_target_id or "",
                action=LEGACY_ACTION.get(resolved.intent, resolved.intent),
                value=legacy_value(resolved.intent, dict(resolved.arguments)),
                owner=resolved.owner,
                place=resolved.area,
            ),
            error=error,
            payload={"trace": trace_text, "status": "rejected"},
        )
