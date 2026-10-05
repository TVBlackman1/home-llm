import logging

from ollama_runner.ha.execute import HaExecutor
from ollama_runner.ha.normalize import Binding
from ollama_runner.request import command_from_result
from ollama_runner.semantic import ResolvedCommand, SemanticCommand, Target
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _snapshot


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self, states: list | None = None, *, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.states = [] if states is None else states
        self.fail = fail

    def get_states(self) -> list:
        return self.states

    def call_service(self, domain: str, service: str, data: dict) -> int:
        if self.fail:
            raise RuntimeError("Authorization: Bearer super-secret-token")
        self.calls.append((domain, service, data))
        return 200


def _state(value: str) -> list:
    return [{"entity_id": "media_player.frame", "state": value, "attributes": {}}]


def _run(intent: str, states: list | None, *, power: str | None = "remote.frame", fail: bool = False):
    inventory = _snapshot()
    bindings = dict(inventory.bindings)
    current = bindings["media_player.frame"]
    bindings["media_player.frame"] = Binding(
        current.entity_id,
        current.domain,
        current.members,
        current.min_color_temp_kelvin,
        current.max_color_temp_kelvin,
        current.color_modes,
        power,
    )
    client = _FakeHa(states, fail=fail)
    result = HaExecutor(inventory.registry, bindings, client).execute(
        ResolvedCommand(
            status="resolved",
            intent=intent,
            semantic_target_id="media_player.frame",
            execution_target_id="media_player.frame",
            arguments={},
            candidates=("media_player.frame",),
        ),
        _Quiet(),
    )
    return client, result


def test_turn_on_from_off_toggles_the_remote():
    client, result = _run("device.turn_on", _state("off"))
    assert client.calls == [("homeassistant", "toggle", {"entity_id": "remote.frame"})]
    assert result.payload["execution"]["current_state"] == "off"
    assert result.payload["execution"]["service"] == "homeassistant.toggle"


def test_turn_on_when_already_on_is_a_noop():
    client, result = _run("device.turn_on", _state("on"))
    command = command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(intent="device.turn_on", target=Target(device_type="tv")),
        resolution_status="resolved",
        device_id="media_player.frame",
    )
    assert client.calls == []
    assert result.payload["execution"]["action"] == "noop"
    assert "service" not in result.payload["execution"]
    assert command.status == "success"
    assert command.executed is False
    assert command.current_state == "on"


def test_turn_off_from_on_toggles_and_already_off_does_not():
    on, _ = _run("device.turn_off", _state("on"))
    off, _ = _run("device.turn_off", _state("off"))
    assert on.calls == [("homeassistant", "toggle", {"entity_id": "remote.frame"})]
    assert off.calls == []


def test_unknown_state_and_missing_remote_do_not_toggle():
    unknown, unknown_result = _run("device.turn_on", _state("unavailable"))
    missing, missing_result = _run("device.turn_off", _state("on"), power=None)
    absent, _ = _run("device.turn_on", [])
    assert unknown.calls == []
    assert unknown_result.payload["reason"] == "tv_power_state_unknown"
    assert missing.calls == []
    assert missing_result.payload["reason"] == "tv_power_unavailable"
    assert absent.calls == []


def test_light_power_still_uses_the_light_service():
    inventory = _snapshot()
    client = _FakeHa()
    HaExecutor(inventory.registry, inventory.bindings, client).execute(
        ResolvedCommand(
            status="resolved",
            intent="device.turn_off",
            semantic_target_id="light.lampa",
            execution_target_id="light.lampa",
            arguments={},
            candidates=("light.lampa",),
        ),
        _Quiet(),
    )
    assert client.calls == [("light", "turn_off", {"entity_id": "light.lampa"})]


def test_toggle_failure_is_execution_failed():
    client, result = _run("device.turn_on", _state("off"), fail=True)
    assert client.calls == []
    assert result.payload["status"] == "execution_failed"
    assert result.payload["reason"] == "execution_exception:RuntimeError"
    assert "super-secret-token" not in result.payload["reason"]


def test_noop_trace_does_not_name_a_service(caplog):
    caplog.set_level(logging.INFO, logger="ollama_runner.request")
    from ollama_runner.request import RequestResult, emit_request

    _client, result = _run("device.turn_on", _state("on"))
    command = command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(intent="device.turn_on", target=Target(device_type="tv")),
        resolution_status="resolved",
        device_id="media_player.frame",
        resolved_type="tv",
    )
    emit_request(RequestResult(1, "Включи телевизор", command.status, (command,)))
    assert "current_state=on" in caplog.text
    assert "action=noop" in caplog.text
    assert "homeassistant.toggle" not in caplog.text
    assert "status=success" in caplog.text
