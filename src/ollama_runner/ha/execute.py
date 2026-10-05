from __future__ import annotations

from dataclasses import dataclass

import httpx

from ollama_runner.ha.normalize import Binding
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.request import EXECUTION_FAILED
from ollama_runner.semantic import ResolvedCommand
from ollama_runner.skills.book import Executor
from ollama_runner.types import Result


_POWER = {
    "device.turn_on": "turn_on",
    "device.turn_off": "turn_off",
}


@dataclass(frozen=True)
class PlannedCall:
    domain: str
    service: str
    entity_id: str


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


def planned_call(intent: str, binding: Binding | None) -> PlannedCall | None:
    """Map a resolved power intent onto the entity domain. Other intents stay unexecuted."""

    service = _POWER.get(intent)
    if service is None or binding is None:
        return None
    return PlannedCall(binding.domain, service, binding.entity_id)


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
        if not self._perform:
            return _with_execution(result, execution)
        try:
            status_code = self._client.call_service(
                call.domain,
                call.service,
                {"entity_id": call.entity_id},
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
