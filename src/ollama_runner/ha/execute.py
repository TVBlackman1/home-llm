from __future__ import annotations

from dataclasses import dataclass

from ollama_runner.ha.normalize import Binding
from ollama_runner.inventory.registry import DeviceRegistry
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
        if self._perform:
            self._client.call_service(
                call.domain,
                call.service,
                {"entity_id": call.entity_id},
            )
        return result
