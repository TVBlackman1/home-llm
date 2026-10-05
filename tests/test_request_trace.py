import logging

import httpx

from ollama_runner.ha.execute import HaExecutor, failure_reason
from ollama_runner.request import redact
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import NotCommandOutcome, SemanticCommand, Target
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _snapshot


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def call_service(self, domain: str, service: str, data: dict) -> int:
        self.calls.append((domain, service, data))
        return 200


class _Boom:
    def call_service(self, domain: str, service: str, data: dict) -> int:
        raise RuntimeError("Authorization: Bearer super-secret-token")


class _FakeNLU:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls = 0

    def parse(self, text: str, context) -> object:
        self.calls += 1
        return self.outcome


def _pipeline(client, *, nlu=None):
    inventory = _snapshot()
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    backend = nlu or _FakeNLU(NotCommandOutcome())
    pipeline = SemanticPipeline(
        backend,
        CapabilityResolver(inventory.registry),
        executor,
        inventory.registry,
    )
    return inventory, client, pipeline


def test_resolved_call_is_success():
    _inventory, client, pipeline = _pipeline(_FakeHa())
    result = pipeline.run("Включи лампу 1", _Quiet(), state=_snapshot().state)
    request = pipeline.last_request
    assert request is not None
    assert result.ok is True
    assert result.payload["status"] == "resolved"
    assert request.status == "success"
    assert request.commands[0].source == "deterministic"
    assert request.commands[0].service == "light.turn_on"
    assert request.commands[0].entity_id == "light.bulb_e27_lemon_3"
    assert client.calls == [("light", "turn_on", {"entity_id": "light.bulb_e27_lemon_3"})]


def test_not_found_does_not_call():
    _inventory, client, pipeline = _pipeline(_FakeHa())
    result = pipeline.run("Включи свет на кухне", _Quiet(), state=_snapshot().state)
    request = pipeline.last_request
    assert request is not None
    assert result.ok is False
    assert result.payload["status"] == "not_found"
    assert request.status == "not_found"
    assert request.commands[0].executed is False
    assert client.calls == []


def test_ambiguous_does_not_call():
    _inventory, client, pipeline = _pipeline(_FakeHa())
    result = pipeline.run("Включи свет в гостиной", _Quiet(), state=_snapshot().state)
    request = pipeline.last_request
    assert request is not None
    assert result.payload["status"] == "ambiguous"
    assert request.status == "ambiguous"
    assert request.commands[0].executed is False
    assert client.calls == []


def test_ha_failure_is_execution_failed(caplog):
    caplog.set_level(logging.DEBUG, logger="ollama_runner.request")
    _inventory, _client, pipeline = _pipeline(_Boom())
    result = pipeline.run("Включи лампу 1", _Quiet(), state=_snapshot().state)
    request = pipeline.last_request
    assert request is not None
    assert result.ok is False
    assert result.payload["status"] == "execution_failed"
    assert request.status == "execution_failed"
    assert request.commands[0].reason == "execution_exception:RuntimeError"
    assert request.commands[0].resolution_status == "resolved"
    assert "super-secret-token" not in caplog.text
    assert "super-secret-token" not in request.commands[0].reason
    assert "Authorization" not in caplog.text


def test_token_shapes_are_redacted():
    leaked = "Authorization: Bearer super-secret-token eyJhbGciOiJI.payload.signature"
    cleaned = redact(leaked)
    assert "super-secret-token" not in cleaned
    assert "eyJhbGciOi" not in cleaned
    assert "authorization [redacted]" in cleaned


def test_failure_reasons_do_not_copy_exception_text():
    assert failure_reason(httpx.ConnectError("Bearer super-secret-token")) == "ha_connection_failed"
    request = httpx.Request("POST", "http://ha.local/api/services/light/turn_on")
    response = httpx.Response(401, request=request)
    error = httpx.HTTPStatusError("Bearer super-secret-token", request=request, response=response)
    assert failure_reason(error) == "ha_authentication_failed"
    response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError("down", request=request, response=response)
    assert failure_reason(error) == "ha_service_failed:500"


def test_compound_keeps_a_result_per_command():
    inventory, client, pipeline = _pipeline(_FakeHa())
    pipeline.run("Выключи свет и телевизор", _Quiet(), state=inventory.state)
    request = pipeline.last_request
    assert request is not None
    assert [command.intent for command in request.commands] == [
        "device.turn_off",
        "device.turn_off",
    ]
    assert [command.device_type for command in request.commands] == ["light", "tv"]
    assert len(client.calls) == sum(command.status == "success" for command in request.commands)


def test_llm_reject_is_visible_and_does_not_call(caplog):
    caplog.set_level(logging.DEBUG, logger="ollama_runner.request")
    nlu = _FakeNLU(NotCommandOutcome(discarded_intent="device.turn_on"))
    _inventory, client, pipeline = _pipeline(_FakeHa(), nlu=nlu)
    result = pipeline.run("привет", _Quiet())
    request = pipeline.last_request
    assert request is not None
    assert nlu.calls == 1
    assert result.payload["status"] == "not_command"
    assert request.status == "not_command"
    assert request.commands[0].source == "llm"
    assert request.commands[0].decision == "not_command"
    assert request.llm is not None
    assert request.llm["discarded_intent"] == "device.turn_on"
    assert request.merge_notes is None
    assert client.calls == []
    assert "[llm]" in caplog.text
    assert "[resolve]" not in caplog.text
    assert "[execute]" not in caplog.text


def test_llm_command_shows_merge(caplog):
    caplog.set_level(logging.INFO, logger="ollama_runner.request")
    command = SemanticCommand(intent="device.turn_on", target=Target(device_type="light"))
    nlu = _FakeNLU(command)
    _inventory, client, pipeline = _pipeline(_FakeHa(), nlu=nlu)
    pipeline.run("сделай потеплее", _Quiet(), state=_snapshot().state)
    request = pipeline.last_request
    assert request is not None
    assert nlu.calls == 1
    assert request.commands[0].source == "llm"
    assert request.merge_notes is not None
    assert "source=llm" in caplog.text
    assert client.calls == [] or request.commands[0].executed is True
