import logging

from ollama_runner.ha.execute import HaExecutor
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.request import RequestResult, command_from_result, emit_request
from ollama_runner.semantic import ResolvedCommand, SemanticCommand, Target
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _snapshot


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self, sources: list[str] | None = None) -> None:
        self.calls: list[tuple] = []
        self.sources = ["TV", "HDMI"] if sources is None else sources

    def get_states(self) -> list:
        attributes = {} if self.sources is None else {"source_list": self.sources}
        return [{"entity_id": "media_player.frame", "state": "on", "attributes": attributes}]

    def call_service(self, domain: str, service: str, data: dict) -> int:
        self.calls.append((domain, service, data))
        return 200


def _run(value: str, *, entity: str = "media_player.frame", status: str = "resolved", sources: list[str] | None = None):
    inventory = _snapshot()
    client = _FakeHa(sources)
    result = HaExecutor(inventory.registry, inventory.bindings, client).execute(
        ResolvedCommand(
            status=status,
            intent="source.select",
            semantic_target_id=entity if status == "resolved" else None,
            execution_target_id=entity if status == "resolved" else None,
            arguments={"value": value} if value else {},
            candidates=(entity,) if status == "resolved" else (),
        ),
        _Quiet(),
    )
    return client, result


def test_switch_phrases_select_the_named_source():
    registry = DeviceRegistry.from_static()
    expected = {
        "Переключи телевизор на HDMI": "HDMI",
        "Переключи телевизор на hdmi": "HDMI",
        "Переключи телевизор на TV": "TV",
        "Переключи телевизор на tv": "TV",
        "Выбери HDMI на телевизоре": "HDMI",
        "Выбери TV на телевизоре": "TV",
    }
    for text, source in expected.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "source.select"
        assert parsed.slots.device_type == "tv"
        assert parsed.value == source
        assert parsed.mention != source


def test_source_rules_leave_power_and_volume_alone():
    registry = DeviceRegistry.from_static()
    expected = {
        "Включи телевизор": ("device.turn_on", "tv"),
        "Выключи телевизор": ("device.turn_off", "tv"),
        "Включи звук на телевизоре": ("volume.unmute", "tv"),
        "Выключи звук на телевизоре": ("volume.mute", "tv"),
        "Сделай телевизор громче": ("volume.increase", "tv"),
        "Включи HDMI на телевизоре": ("device.turn_on", "tv"),
        "Поставь HDMI на телевизоре": ("device.turn_on", "tv"),
    }
    numbered = parse_deterministic("Переключи телевизор на HDMI 1", registry)
    assert numbered.intent == "source.select"
    assert numbered.value == "HDMI 1"
    for text, (intent, device_type) in expected.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == intent
        assert parsed.slots.device_type == device_type


def test_select_source_uses_the_advertised_name():
    inventory = _snapshot()
    assert "source.select" in inventory.registry.get("media_player.frame").capabilities
    hdmi, _ = _run("HDMI")
    tv, _ = _run("tv")
    assert hdmi.calls == [
        ("media_player", "select_source", {"entity_id": "media_player.frame", "source": "HDMI"})
    ]
    assert tv.calls == [
        ("media_player", "select_source", {"entity_id": "media_player.frame", "source": "TV"})
    ]
    assert "remote" not in str(hdmi.calls)
    assert "toggle" not in str(hdmi.calls)


def test_missing_source_attribute_still_selects_from_the_list(caplog):
    caplog.set_level(logging.INFO, logger="ollama_runner.request")
    client, result = _run("HDMI")
    assert "source" not in client.get_states()[0]["attributes"]
    assert client.calls == [
        ("media_player", "select_source", {"entity_id": "media_player.frame", "source": "HDMI"})
    ]
    assert "source_changed" not in result.payload["execution"]
    command = command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(
            intent="source.select",
            target=Target(device_type="tv", mention="телевизор"),
            arguments={"value": "HDMI"},
        ),
        resolution_status="resolved",
        device_id="media_player.frame",
        resolved_type="tv",
    )
    assert command.status == "success"
    assert command.executed is True
    emit_request(RequestResult(1, "Переключи телевизор на HDMI", command.status, (command,)))
    assert "intent=source.select" in caplog.text
    assert "service=media_player.select_source" in caplog.text
    assert "entity=media_player.frame" in caplog.text
    assert "source=HDMI" in caplog.text
    assert "status=success" in caplog.text
    assert "source_changed" not in caplog.text
    again, _ = _run("HDMI")
    assert len(again.calls) == 1


def test_unknown_or_unreadable_source_is_not_sent():
    unknown, unknown_result = _run("Chromecast")
    numbered, numbered_result = _run("HDMI 1")
    missing, missing_result = _run("HDMI", sources=[])
    assert unknown.calls == []
    assert numbered.calls == []
    assert missing.calls == []
    assert unknown_result.payload["reason"] == "source_unavailable"
    assert numbered_result.payload["reason"] == "source_unavailable"
    assert missing_result.payload["reason"] == "source_unavailable"


def test_source_is_not_sent_without_a_target():
    light, light_result = _run("HDMI", entity="light.lampa")
    missing, _ = _run("HDMI", status="not_found")
    ambiguous, _ = _run("HDMI", status="ambiguous")
    assert light.calls == []
    assert missing.calls == []
    assert ambiguous.calls == []
    assert light_result.error == "execution target lacks capability"
