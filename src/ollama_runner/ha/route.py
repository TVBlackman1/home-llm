"""Dispatch a resolved semantic command to the backend that implements it."""

from __future__ import annotations

from ollama_runner.execution import ExecutionBackend, ExecutionBinding, execution_capability
from ollama_runner.ha.execute import HaExecutor
from ollama_runner.ha.normalize import Binding
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.semantic import ResolvedCommand
from ollama_runner.types import Result


class ExecutionRouter:
    """Implementation and backend live here. Home Assistant details do not.

    The resolver has already chosen a semantic device. This chooses whether
    that intent is implemented and which executor receives it.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        bindings: dict[str, Binding],
        client,
        *,
        perform: bool = True,
    ) -> None:
        self._bindings = bindings
        self._ha = HaExecutor(registry, bindings, client, perform=perform)
        self._executors = {ExecutionBackend.HOME_ASSISTANT: self._ha}

    def execute(self, resolved: ResolvedCommand, sink) -> Result:
        if resolved.status != "resolved" or not resolved.execution_target_id:
            return self._ha.execute(resolved, sink)
        capability = execution_capability(resolved.intent)
        if capability is None:
            return self._ha.execute(resolved, sink)
        binding = self._bindings.get(resolved.execution_target_id)
        if binding is None:
            return self._ha.execute(resolved, sink)
        spec = ExecutionBinding(capability.backend, binding.for_capability(resolved.intent))
        executor = self._executors.get(spec.backend)
        if executor is None:
            return self._ha.execute(resolved, sink)
        return executor.execute(resolved, sink)
