from __future__ import annotations

from dataclasses import dataclass

import httpx

from ollama_runner.ha.normalize import Binding
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.values import percent_points
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


@dataclass(frozen=True)
class PlannedCall:
    domain: str
    service: str
    entity_id: str
    brightness_pct: int | None = None
    brightness_step_pct: int | None = None

    def service_data(self) -> dict:
        data = {"entity_id": self.entity_id}
        if self.brightness_pct is not None:
            data["brightness_pct"] = self.brightness_pct
        if self.brightness_step_pct is not None:
            data["brightness_step_pct"] = self.brightness_step_pct
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
    """Run the existing executor, then call Home Assistant only for a resolved power command."""

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
        call = planned_call(
            resolved.intent,
            self._bindings.get(resolved.execution_target_id or ""),
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
