from __future__ import annotations

from dataclasses import dataclass

import httpx

from ollama_runner.ha.normalize import Binding, whole_kelvin
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.values import kelvin_points, percent_points
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


@dataclass(frozen=True)
class PlannedCall:
    domain: str
    service: str
    entity_id: str
    brightness_pct: int | None = None
    brightness_step_pct: int | None = None
    color_temp_kelvin: int | None = None

    def service_data(self) -> dict:
        data = {"entity_id": self.entity_id}
        if self.brightness_pct is not None:
            data["brightness_pct"] = self.brightness_pct
        if self.brightness_step_pct is not None:
            data["brightness_step_pct"] = self.brightness_step_pct
        if self.color_temp_kelvin is not None:
            data["color_temp_kelvin"] = self.color_temp_kelvin
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
    service = _POWER.get(intent)
    if service is not None:
        return PlannedCall(binding.domain, service, binding.entity_id)
    if intent not in _BRIGHTNESS or binding.domain != "light":
        return None
    points = percent_points(value)
    if intent == "brightness.set":
        if points is None or not 0 <= points <= 100:
            return None
        return PlannedCall(binding.domain, "turn_on", binding.entity_id, brightness_pct=points)
    if value.strip() and points is None:
        return None
    step = DEFAULT_BRIGHTNESS_STEP if points is None else points
    if not 1 <= step <= 100:
        return None
    signed = step if intent == "brightness.increase" else -step
    return PlannedCall(
        binding.domain,
        "turn_on",
        binding.entity_id,
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
        binding = self._bindings.get(resolved.execution_target_id or "")
        relative: dict = {}
        if resolved.intent in _COLOR_TEMPERATURE:
            current = None
            if (
                binding is not None
                and resolved.intent in {"color_temperature.warmer", "color_temperature.cooler"}
            ):
                current = self._current_kelvin(binding.entity_id)
            call, reason, relative = _color_temperature_call(
                resolved.intent,
                binding,
                str(resolved.arguments.get("value") or ""),
                current,
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

    if binding is None or binding.domain != "light":
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
            PlannedCall(binding.domain, "turn_on", binding.entity_id, color_temp_kelvin=kelvin),
            "",
            {},
        )
    if current is None:
        return None, "color_temp_unavailable", {}
    direction = "warmer" if intent == "color_temperature.warmer" else "cooler"
    step = DEFAULT_COLOR_TEMP_STEP_K
    raw = current - step if direction == "warmer" else current + step
    target = min(max(raw, low), high)
    return (
        PlannedCall(binding.domain, "turn_on", binding.entity_id, color_temp_kelvin=target),
        "",
        {
            "current_kelvin": current,
            "direction": direction,
            "step_kelvin": step,
            "target_kelvin": target,
        },
    )


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
