import logging

from ollama_runner.ha.execute import HaExecutor
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.request import RequestResult, command_from_result, emit_request
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import NotCommandOutcome, ResolvedCommand, SemanticCommand, Target
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _snapshot


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.reads = 0
        self.fail = fail

    def get_states(self) -> list:
        self.reads += 1
        return [{
            "entity_id": "media_player.frame",
            "state": "on",
            "attributes": {"supported_features": 24509, "source_list": ["TV", "HDMI"]},
        }]

    def call_service(self, domain: str, service: str, data: dict) -> int:
        if self.fail:
            raise RuntimeError("Authorization: Bearer super-secret-token")
        self.calls.append((domain, service, data))
        return 200


def _run(intent: str, *, entity: str = "media_player.frame", status: str = "resolved", fail: bool = False):
    inventory = _snapshot()
    client = _FakeHa(fail=fail)
    result = HaExecutor(inventory.registry, inventory.bindings, client).execute(
        ResolvedCommand(
            status=status,
            intent=intent,
            semantic_target_id=entity if status == "resolved" else None,
            execution_target_id=entity if status == "resolved" else None,
            arguments={},
            candidates=(entity,) if status == "resolved" else (),
        ),
        _Quiet(),
    )
    return client, result


def test_resume_and_pause_phrases():
    registry = DeviceRegistry.from_static()
    resumed = {
        "Продолжи": (None, None),
        "Продолжи на телевизоре": ("tv", "телевизоре"),
        "Продолжи воспроизведение": (None, None),
    }
    for text, (device_type, mention) in resumed.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "media.resume"
        assert parsed.slots.device_type == device_type
        assert parsed.mention == mention
        assert parsed.value is None
    paused = {
        "Поставь на паузу": None,
        "Поставь телевизор на паузу": "tv",
        "Приостанови телевизор": "tv",
        "Поставь сериал на паузу": None,
    }
    for text, device_type in paused.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "media.pause"
        assert parsed.slots.device_type == device_type
    unchanged = {
        "Включи телевизор": "device.turn_on",
        "Выключи телевизор": "device.turn_off",
        "Сделай телевизор громче": "volume.increase",
        "Поставь громкость телевизора на 30 процентов": "volume.set",
        "Выключи звук на телевизоре": "volume.mute",
        "Включи звук на телевизоре": "volume.unmute",
        "Переключи телевизор на HDMI": "source.select",
        "Переключи телевизор на TV": "source.select",
        "Останови телевизор": "media.stop",
    }
    for text, intent in unchanged.items():
        assert parse_deterministic(text, registry).intent == intent


def test_play_and_pause_use_the_media_player():
    inventory = _snapshot()
    capabilities = inventory.registry.get("media_player.frame").capabilities
    assert "media.pause" in capabilities
    assert "media.play" in capabilities
    assert "media.resume" in capabilities
    pause, _ = _run("media.pause")
    play, _ = _run("media.play")
    resume, _ = _run("media.resume")
    assert pause.reads == 0
    assert pause.calls == [("media_player", "media_pause", {"entity_id": "media_player.frame"})]
    assert play.calls == [("media_player", "media_play", {"entity_id": "media_player.frame"})]
    assert resume.calls == [("media_player", "media_play", {"entity_id": "media_player.frame"})]
    for client in (pause, play, resume):
        rendered = str(client.calls)
        assert "remote" not in rendered
        assert "toggle" not in rendered
        assert "volume" not in rendered
        assert "source" not in rendered


def test_playback_runs_without_a_readable_session(caplog):
    caplog.set_level(logging.INFO, logger="ollama_runner.request")
    client, result = _run("media.pause")
    state = client.get_states()[0]
    assert state["state"] == "on"
    assert "playing" not in state["attributes"]
    assert "paused" not in state["attributes"]
    assert "media_title" not in state["attributes"]
    assert client.calls[0][1] == "media_pause"
    command = command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(
            intent="media.pause",
            target=Target(device_type="tv", mention="телевизор"),
        ),
        resolution_status="resolved",
        device_id="media_player.frame",
        resolved_type="tv",
    )
    assert command.status == "success"
    emit_request(RequestResult(1, "Поставь телевизор на паузу", command.status, (command,)))
    assert "service=media_player.media_pause" in caplog.text
    assert "entity=media_player.frame" in caplog.text
    assert "status=success" in caplog.text
    assert "playing" not in caplog.text
    assert "paused" not in caplog.text


def test_playback_is_not_sent_without_a_target():
    light, light_result = _run("media.pause", entity="light.lampa")
    missing, _ = _run("media.resume", status="not_found")
    ambiguous, _ = _run("media.play", status="ambiguous")
    assert light.calls == []
    assert missing.calls == []
    assert ambiguous.calls == []
    assert light_result.error == "execution target lacks capability"


def test_playback_failure_is_execution_failed():
    client, result = _run("media.resume", fail=True)
    assert client.calls == []
    assert result.payload["status"] == "execution_failed"
    assert "super-secret-token" not in result.payload["reason"]


def _recorded(result, intent: str):
    return command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(intent=intent, target=Target(device_type="tv", mention="телевизор")),
        resolution_status="resolved",
        device_id="media_player.frame",
        resolved_type="tv",
    )


def test_stop_is_advertised_and_not_executed():
    inventory = _snapshot()
    assert "media.stop" in inventory.registry.get("media_player.frame").capabilities
    client, result = _run("media.stop")
    command = _recorded(result, "media.stop")
    assert client.calls == []
    assert result.ok is False
    assert result.payload["status"] == "unsupported"
    assert result.payload["reason"] == "execution_not_implemented"
    assert "service" not in (result.payload.get("execution") or {})
    assert command.status == "unsupported"
    assert command.executed is False
    assert command.service is None
    assert command.action is None


def test_next_is_advertised_and_not_executed():
    inventory = _snapshot()
    assert "media.next" in inventory.registry.get("media_player.frame").capabilities
    client, result = _run("media.next")
    command = _recorded(result, "media.next")
    assert client.calls == []
    assert result.payload["status"] == "unsupported"
    assert result.payload["reason"] == "execution_not_implemented"
    assert command.status == "unsupported"
    assert command.status != "success"


def test_stop_request_is_not_success(caplog):
    caplog.set_level(logging.INFO, logger="ollama_runner.request")
    inventory = _snapshot()
    client = _FakeHa()

    class _Idle:
        calls = 0

        def parse(self, text, context):
            self.calls += 1
            return NotCommandOutcome()

    nlu = _Idle()
    pipeline = SemanticPipeline(
        nlu,
        CapabilityResolver(inventory.registry),
        HaExecutor(inventory.registry, inventory.bindings, client),
        inventory.registry,
    )
    result = pipeline.run("Останови телевизор", _Quiet(), state=inventory.state)
    request = pipeline.last_request
    assert request is not None
    assert nlu.calls == 0
    assert client.calls == []
    assert result.ok is False
    assert request.commands[0].resolution_status == "resolved"
    assert request.commands[0].intent == "media.stop"
    assert request.status == "unsupported"
    assert request.commands[0].reason == "execution_not_implemented"
    assert request.commands[0].executed is False
    assert "media_player.media_stop" not in caplog.text
    assert "service_called=false" in caplog.text
    assert "status=unsupported" in caplog.text
    assert "reason=execution_not_implemented" in caplog.text


def test_matching_power_noop_stays_success():
    client, result = _run("device.turn_on")
    command = _recorded(result, "device.turn_on")
    assert client.calls == []
    assert result.payload["execution"]["action"] == "noop"
    assert "service" not in result.payload["execution"]
    assert command.status == "success"
    assert command.executed is False
    assert command.action == "noop"
    assert command.reason != "execution_not_implemented"
