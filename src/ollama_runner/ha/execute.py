from __future__ import annotations

from dataclasses import dataclass

import httpx

from ollama_runner.execution import execution_capability
from ollama_runner.ha.normalize import Binding, whole_kelvin
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.values import kelvin_points, named_color_rgb, percent_points
from ollama_runner.request import EXECUTION_FAILED
from ollama_runner.semantic import ResolvedCommand
from ollama_runner.skills.book import Executor
from ollama_runner.types import Result


_POWER = {
    "device.turn_on": "turn_on",
    "device.turn_off": "turn_off",
}
_BRIGHTNESS = frozenset({
    "brightness.set",
    "brightness.increase",
    "brightness.decrease",
})
# No step exists in the semantic layer. Ten points on the 0–100 service scale
# is the one explicit default for a relative command that names no amount.
DEFAULT_BRIGHTNESS_STEP = 10
_COLOR_TEMPERATURE = frozenset({
    "color_temperature.set",
    "color_temperature.warmer",
    "color_temperature.cooler",
})
# The installed bulbs span 2700–6500 K. 400 K is one step inside the
# 300–500 K band: visible on that 3800 K range, and not a large jump.
DEFAULT_COLOR_TEMP_STEP_K = 400
# light.turn_on has no hs_color field. rgb_color is the field filtered for hs.
_CHROMATIC_MODES = frozenset({"hs", "xy", "rgb", "rgbw", "rgbww"})
_PLAYBACK = {
    "media.pause": "media_pause",
    "media.play": "media_play",
    "media.resume": "media_play",
}
_VOLUME = frozenset({
    "volume.set",
    "volume.increase",
    "volume.decrease",
    "volume.mute",
    "volume.unmute",
})


@dataclass(frozen=True)
class PlannedCall:
    domain: str
    service: str
    entity_id: str
    brightness_pct: int | None = None
    brightness_step_pct: int | None = None
    color_temp_kelvin: int | None = None
    rgb_color: tuple[int, int, int] | None = None
    volume_level: float | None = None
    is_volume_muted: bool | None = None
    source: str | None = None

    def service_data(self) -> dict:
        data = {"entity_id": self.entity_id}
        if self.brightness_pct is not None:
            data["brightness_pct"] = self.brightness_pct
        if self.brightness_step_pct is not None:
            data["brightness_step_pct"] = self.brightness_step_pct
        if self.color_temp_kelvin is not None:
            data["color_temp_kelvin"] = self.color_temp_kelvin
        if self.rgb_color is not None:
            data["rgb_color"] = list(self.rgb_color)
        if self.volume_level is not None:
            data["volume_level"] = self.volume_level
        if self.is_volume_muted is not None:
            data["is_volume_muted"] = self.is_volume_muted
        if self.source is not None:
            data["source"] = self.source
        return data


def failure_reason(exc: BaseException) -> str:
    """Stable reason code. The exception text is never copied: it can hold a token."""

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in {401, 403}:
            return "ha_authentication_failed"
        return f"ha_service_failed:{code}"
    if isinstance(exc, httpx.TransportError):
        return "ha_connection_failed"
    return f"execution_exception:{type(exc).__name__}"


def planned_call(
    intent: str,
    binding: Binding | None,
    value: str = "",
) -> PlannedCall | None:
    """Map a resolved power or light-brightness intent. Other intents stay unexecuted."""

    if binding is None:
        return None
    target = binding.for_capability(intent)
    service = _POWER.get(intent)
    if service is not None:
        return PlannedCall(target.domain, service, target.entity_id)
    if intent not in _BRIGHTNESS or target.domain != "light":
        return None
    points = percent_points(value)
    if intent == "brightness.set":
        if points is None or not 0 <= points <= 100:
            return None
        return PlannedCall(target.domain, "turn_on", target.entity_id, brightness_pct=points)
    if value.strip() and points is None:
        return None
    step = DEFAULT_BRIGHTNESS_STEP if points is None else points
    if not 1 <= step <= 100:
        return None
    signed = step if intent == "brightness.increase" else -step
    return PlannedCall(
        target.domain,
        "turn_on",
        target.entity_id,
        brightness_step_pct=signed,
    )


class HaExecutor:
    """Run the existing executor, then call Home Assistant for a mapped light command."""

    def __init__(
        self,
        registry: DeviceRegistry,
        bindings: dict[str, Binding],
        client,
        *,
        perform: bool = True,
    ) -> None:
        self._inner = Executor(registry)
        self._bindings = bindings
        self._client = client
        self._perform = perform
        self.planned: list[PlannedCall] = []

    def execute(self, resolved: ResolvedCommand, sink) -> Result:
        result = self._inner.execute(resolved, sink)
        if not result.ok or resolved.status != "resolved":
            return result
        if execution_capability(resolved.intent) is None:
            return _unsupported(result, "execution_not_implemented")
        binding = self._bindings.get(resolved.execution_target_id or "")
        if binding is None:
            return _unsupported(result, "binding_unavailable")
        target = binding.for_capability(resolved.intent)
        relative: dict = {}
        if (
            resolved.intent in _POWER
            and binding is not None
            and binding.default.domain == "media_player"
        ):
            return self._toggle_power(resolved, result, binding)
        if resolved.intent in _VOLUME:
            if target is None or target.domain != "media_player":
                return _unsupported(result, "volume_unavailable")
            return self._change_volume(resolved, result, binding)
        if resolved.intent == "source.select":
            if target is None or target.domain != "media_player":
                return _unsupported(result, "source_unavailable")
            return self._select_source(resolved, result, binding)
        if resolved.intent in _PLAYBACK:
            if target is None or target.domain != "media_player":
                return _unsupported(result, "playback_unavailable")
            return self._control_playback(resolved, result, binding)
        if resolved.intent in _COLOR_TEMPERATURE:
            current = None
            if (
                target is not None
                and resolved.intent in {"color_temperature.warmer", "color_temperature.cooler"}
            ):
                current = self._current_kelvin(target.entity_id)
            call, reason, relative = _color_temperature_call(
                resolved.intent,
                binding,
                str(resolved.arguments.get("value") or ""),
                current,
            )
            if call is None:
                return _unsupported(result, reason)
        elif resolved.intent == "color.set":
            call, reason = _named_color_call(
                binding,
                str(resolved.arguments.get("value") or ""),
            )
            if call is None:
                return _unsupported(result, reason)
        else:
            call = planned_call(
                resolved.intent,
                binding,
                str(resolved.arguments.get("value") or ""),
            )
        if call is None:
            return result
        self.planned.append(call)
        service = f"{call.domain}.{call.service}"
        execution = {
            "attempted": self._perform,
            "service": service,
            "entity_id": call.entity_id,
        }
        if call.brightness_pct is not None:
            execution["brightness_pct"] = call.brightness_pct
        if call.brightness_step_pct is not None:
            execution["brightness_step_pct"] = call.brightness_step_pct
        if call.color_temp_kelvin is not None:
            execution["color_temp_kelvin"] = call.color_temp_kelvin
        if call.rgb_color is not None:
            execution["rgb_color"] = list(call.rgb_color)
        if call.volume_level is not None:
            execution["volume_level"] = call.volume_level
        execution.update(relative)
        if not self._perform:
            return _with_execution(result, execution)
        try:
            status_code = self._client.call_service(
                call.domain,
                call.service,
                call.service_data(),
            )
        except Exception as exc:
            execution["attempted"] = True
            return _failed(result, failure_reason(exc), execution)
        if isinstance(status_code, int):
            execution["http_status"] = status_code
        return _with_execution(result, execution)

    def _change_volume(self, resolved: ResolvedCommand, result: Result, binding: Binding) -> Result:
        """Volume stays on the media player. An explicit percent is not clamped; a relative step is."""

        value = str(resolved.arguments.get("value") or "")
        points = percent_points(value)
        intent = resolved.intent
        entity_id = binding.for_capability(intent).entity_id
        relative: dict = {}
        if intent in {"volume.mute", "volume.unmute"}:
            call = PlannedCall(
                "media_player",
                "volume_mute",
                entity_id,
                is_volume_muted=intent == "volume.mute",
            )
        elif intent == "volume.set":
            if points is None or not 0 <= points <= 100:
                return _unsupported(result, "volume_out_of_range")
            call = PlannedCall(
                "media_player",
                "volume_set",
                entity_id,
                volume_level=_percent_level(points),
            )
        elif value.strip() and points is None:
            return _unsupported(result, "volume_out_of_range")
        elif points is None:
            service = "volume_up" if intent == "volume.increase" else "volume_down"
            call = PlannedCall("media_player", service, entity_id)
        else:
            if not 1 <= points <= 100:
                return _unsupported(result, "volume_out_of_range")
            current = self._volume_level(entity_id)
            if current is None:
                return _unsupported(result, "volume_unavailable")
            signed = points if intent == "volume.increase" else -points
            target = min(100, max(0, round(current * 100) + signed))
            level = _percent_level(target)
            relative = {
                "current_volume": round(current, 2),
                "delta_percent": signed,
                "target_volume": level,
            }
            call = PlannedCall("media_player", "volume_set", entity_id, volume_level=level)
        self.planned.append(call)
        execution = {
            "attempted": self._perform,
            "intent": intent,
            "service": f"{call.domain}.{call.service}",
            "entity_id": call.entity_id,
        }
        if call.volume_level is not None:
            execution["volume_level"] = call.volume_level
        if call.is_volume_muted is not None:
            execution["is_volume_muted"] = call.is_volume_muted
        execution.update(relative)
        if not self._perform:
            return _with_execution(result, execution)
        try:
            status_code = self._client.call_service(call.domain, call.service, call.service_data())
        except Exception as exc:
            execution["attempted"] = True
            return _failed(result, failure_reason(exc), execution)
        if isinstance(status_code, int):
            execution["http_status"] = status_code
        return _with_execution(result, execution)

    def _control_playback(self, resolved: ResolvedCommand, result: Result, binding: Binding) -> Result:
        """Play and pause use the media player. Stop stays unwired until it is verified."""

        call = PlannedCall(
            "media_player",
            _PLAYBACK[resolved.intent],
            binding.for_capability(resolved.intent).entity_id,
        )
        self.planned.append(call)
        execution = {
            "attempted": self._perform,
            "intent": resolved.intent,
            "service": f"media_player.{call.service}",
            "entity_id": call.entity_id,
        }
        if not self._perform:
            return _with_execution(result, execution)
        try:
            status_code = self._client.call_service(call.domain, call.service, call.service_data())
        except Exception as exc:
            execution["attempted"] = True
            return _failed(result, failure_reason(exc), execution)
        if isinstance(status_code, int):
            execution["http_status"] = status_code
        return _with_execution(result, execution)

    def _select_source(self, resolved: ResolvedCommand, result: Result, binding: Binding) -> Result:
        """Select a source the media player advertises. No power and no nearest-name guess."""

        entity_id = binding.for_capability(resolved.intent).entity_id
        requested = str(resolved.arguments.get("value") or "").strip()
        chosen = _listed_source(requested, self._source_list(entity_id))
        if chosen is None:
            return _unsupported(result, "source_unavailable")
        call = PlannedCall("media_player", "select_source", entity_id, source=chosen)
        self.planned.append(call)
        execution = {
            "attempted": self._perform,
            "intent": resolved.intent,
            "service": "media_player.select_source",
            "entity_id": call.entity_id,
            "source": chosen,
        }
        if not self._perform:
            return _with_execution(result, execution)
        try:
            status_code = self._client.call_service(call.domain, call.service, call.service_data())
        except Exception as exc:
            execution["attempted"] = True
            return _failed(result, failure_reason(exc), execution)
        if isinstance(status_code, int):
            execution["http_status"] = status_code
        return _with_execution(result, execution)

    def _source_list(self, entity_id: str) -> list[str] | None:
        try:
            states = self._client.get_states()
        except Exception:
            return None
        if not isinstance(states, list):
            return None
        for row in states:
            if not isinstance(row, dict) or row.get("entity_id") != entity_id:
                continue
            attributes = row.get("attributes") or {}
            if not isinstance(attributes, dict):
                return None
            raw = attributes.get("source_list")
            if not isinstance(raw, list):
                return None
            names = [item for item in raw if isinstance(item, str) and item.strip()]
            return names or None
        return None

    def _volume_level(self, entity_id: str) -> float | None:
        try:
            states = self._client.get_states()
        except Exception:
            return None
        if not isinstance(states, list):
            return None
        for row in states:
            if not isinstance(row, dict) or row.get("entity_id") != entity_id:
                continue
            attributes = row.get("attributes") or {}
            if not isinstance(attributes, dict):
                return None
            raw = attributes.get("volume_level")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            level = float(raw)
            if not 0 <= level <= 1:
                return None
            return level
        return None

    def _toggle_power(self, resolved: ResolvedCommand, result: Result, binding: Binding) -> Result:
        """Desired-state power. State comes from the media player; the call uses the power binding."""

        current = self._reported_power(binding.default.entity_id)
        if current not in {"on", "off"}:
            return _unsupported(result, "tv_power_state_unknown")
        want_on = resolved.intent == "device.turn_on"
        if (want_on and current == "on") or (not want_on and current == "off"):
            return _with_execution(
                result,
                {
                    "attempted": False,
                    "noop": True,
                    "intent": resolved.intent,
                    "current_state": current,
                    "action": "noop",
                },
            )
        power = binding.override(resolved.intent)
        if power is None:
            return _unsupported(result, "tv_power_unavailable")
        call = PlannedCall("homeassistant", "toggle", power.entity_id)
        self.planned.append(call)
        execution = {
            "attempted": self._perform,
            "intent": resolved.intent,
            "current_state": current,
            "service": "homeassistant.toggle",
            "entity_id": power.entity_id,
        }
        if not self._perform:
            return _with_execution(result, execution)
        try:
            status_code = self._client.call_service(
                call.domain,
                call.service,
                call.service_data(),
            )
        except Exception as exc:
            execution["attempted"] = True
            return _failed(result, failure_reason(exc), execution)
        if isinstance(status_code, int):
            execution["http_status"] = status_code
        return _with_execution(result, execution)

    def _reported_power(self, entity_id: str) -> str | None:
        """Media-player on/off only. Anything else is not a reason to toggle."""

        try:
            states = self._client.get_states()
        except Exception:
            return None
        if not isinstance(states, list):
            return None
        for row in states:
            if not isinstance(row, dict) or row.get("entity_id") != entity_id:
                continue
            state = row.get("state")
            if state in {"on", "off"}:
                return state
            return None
        return None

    def _current_kelvin(self, entity_id: str) -> int | None:
        """One state read for one relative command. Nothing is cached or retried."""

        try:
            states = self._client.get_states()
        except Exception:
            return None
        if not isinstance(states, list):
            return None
        for row in states:
            if not isinstance(row, dict) or row.get("entity_id") != entity_id:
                continue
            attributes = row.get("attributes") or {}
            if not isinstance(attributes, dict):
                return None
            return whole_kelvin(attributes.get("color_temp_kelvin"))
        return None


def _color_temperature_call(
    intent: str,
    binding: Binding | None,
    value: str,
    current: int | None,
) -> tuple[PlannedCall | None, str, dict]:
    """Absolute Kelvin is rejected outside the device range. Relative movement clamps to it.

    ``light.turn_on`` with ``color_temp_kelvin`` turns an off light on. That is
    the service behavior for an explicit temperature, same as brightness.
    """

    if binding is None:
        return None, "color_temp_unavailable", {}
    target = binding.for_capability(intent)
    if target.domain != "light":
        return None, "color_temp_unavailable", {}
    low = binding.min_color_temp_kelvin
    high = binding.max_color_temp_kelvin
    if low is None or high is None:
        return None, "color_temp_unavailable", {}
    if intent == "color_temperature.set":
        kelvin = kelvin_points(value)
        if kelvin is None or kelvin < low or kelvin > high:
            return None, "color_temp_out_of_range", {}
        return (
            PlannedCall(target.domain, "turn_on", target.entity_id, color_temp_kelvin=kelvin),
            "",
            {},
        )
    if current is None:
        return None, "color_temp_unavailable", {}
    direction = "warmer" if intent == "color_temperature.warmer" else "cooler"
    step = DEFAULT_COLOR_TEMP_STEP_K
    raw = current - step if direction == "warmer" else current + step
    kelvin_target = min(max(raw, low), high)
    return (
        PlannedCall(target.domain, "turn_on", target.entity_id, color_temp_kelvin=kelvin_target),
        "",
        {
            "current_kelvin": current,
            "direction": direction,
            "step_kelvin": step,
            "target_kelvin": kelvin_target,
        },
    )


def _listed_source(requested: str, available: list[str] | None) -> str | None:
    """Case-only match against the device's own source_list."""

    if not requested or not available:
        return None
    folded = requested.casefold()
    for name in available:
        if name.casefold() == folded:
            return name
    return None


def _percent_level(points: int) -> float:
    return round(points / 100, 2)


def _named_color_call(binding: Binding | None, value: str) -> tuple[PlannedCall | None, str]:
    """Named color.set becomes rgb_color. It does not send Kelvin or brightness.

    ``light.turn_on`` with a color turns an off light on, same as brightness
    and color temperature. A chromatic name is not rewritten as Kelvin.
    """

    if binding is None:
        return None, "color_unavailable"
    target = binding.for_capability("color.set")
    if target.domain != "light":
        return None, "color_unavailable"
    rgb = named_color_rgb(value)
    if rgb is None:
        return None, "color_unsupported"
    if not binding.color_modes & _CHROMATIC_MODES:
        return None, "color_unavailable"
    return PlannedCall(target.domain, "turn_on", target.entity_id, rgb_color=rgb), ""


def _unsupported(result: Result, reason: str) -> Result:
    payload = dict(result.payload or {})
    payload["status"] = "unsupported"
    payload["reason"] = reason
    return Result(
        ok=False,
        command=result.command,
        error="unsupported",
        payload=payload,
    )


def _with_execution(result: Result, execution: dict) -> Result:
    payload = dict(result.payload or {})
    payload["execution"] = execution
    return Result(
        ok=result.ok,
        command=result.command,
        error=result.error,
        payload=payload,
    )


def _failed(result: Result, reason: str, execution: dict) -> Result:
    payload = dict(result.payload or {})
    payload["status"] = EXECUTION_FAILED
    payload["reason"] = reason
    payload["execution"] = execution
    return Result(
        ok=False,
        command=result.command,
        error=EXECUTION_FAILED,
        payload=payload,
    )
