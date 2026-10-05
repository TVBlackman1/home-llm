import logging

from ollama_runner.ha.execute import HaExecutor
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.nlu.values import percent_points
from ollama_runner.request import RequestResult, command_from_result, emit_request
from ollama_runner.semantic import ResolvedCommand, SemanticCommand, Target
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _snapshot


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self, level: float | None = None, *, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.level = level
        self.fail = fail
        self.reads = 0

    def get_states(self) -> list:
        self.reads += 1
        attributes = {} if self.level is None else {"volume_level": self.level}
        return [{"entity_id": "media_player.frame", "state": "on", "attributes": attributes}]

    def call_service(self, domain: str, service: str, data: dict) -> int:
        if self.fail:
            raise RuntimeError("Authorization: Bearer super-secret-token")
        self.calls.append((domain, service, data))
        return 200


def _run(intent: str, value: str = "", *, entity: str = "media_player.frame", level: float | None = None, status: str = "resolved", fail: bool = False):
    inventory = _snapshot()
    client = _FakeHa(level, fail=fail)
    result = HaExecutor(inventory.registry, inventory.bindings, client).execute(
        ResolvedCommand(
            status=status,
            intent=intent,
            semantic_target_id=entity if status == "resolved" else None,
            execution_target_id=entity if status == "resolved" else None,
            arguments={"value": value} if value else {},
            candidates=(entity,) if status == "resolved" else (),
        ),
        _Quiet(),
    )
    return client, result


def test_absolute_volume_maps_percent_to_level():
    thirty, _ = _run("volume.set", "на 30 процентов")
    zero, _ = _run("volume.set", "0 процентов")
    full, _ = _run("volume.set", "100")
    assert thirty.reads == 0
    assert thirty.calls == [
        ("media_player", "volume_set", {"entity_id": "media_player.frame", "volume_level": 0.3})
    ]
    assert zero.calls[0][2]["volume_level"] == 0.0
    assert full.calls[0][2]["volume_level"] == 1.0
    assert "remote" not in str(thirty.calls)


def test_invalid_absolute_volume_is_not_sent():
    for value in ("150 процентов", "", "немного"):
        client, result = _run("volume.set", value)
        assert client.calls == []
        assert result.payload["reason"] == "volume_out_of_range"


def test_direction_only_uses_native_relative_services():
    up, _ = _run("volume.increase")
    down, _ = _run("volume.decrease")
    assert up.reads == 0
    assert down.reads == 0
    assert up.calls == [("media_player", "volume_up", {"entity_id": "media_player.frame"})]
    assert down.calls == [("media_player", "volume_down", {"entity_id": "media_player.frame"})]


def test_explicit_delta_sets_the_clamped_target():
    up, up_result = _run("volume.increase", "на 10 процентов", level=0.4)
    down, _ = _run("volume.decrease", "на 10 процентов", level=0.4)
    high, _ = _run("volume.increase", "на 10 процентов", level=0.95)
    low, _ = _run("volume.decrease", "на 10 процентов", level=0.05)
    assert up.calls == [
        ("media_player", "volume_set", {"entity_id": "media_player.frame", "volume_level": 0.5})
    ]
    assert down.calls[0][2]["volume_level"] == 0.3
    assert high.calls[0][2]["volume_level"] == 1.0
    assert low.calls[0][2]["volume_level"] == 0.0
    assert up_result.payload["execution"]["delta_percent"] == 10
    assert up_result.payload["execution"]["current_volume"] == 0.4
    assert up_result.payload["execution"]["target_volume"] == 0.5


def test_volume_does_not_use_the_remote_or_other_targets():
    light, _ = _run("volume.set", "30 процентов", entity="light.lampa")
    missing, _ = _run("volume.set", "30 процентов", status="not_found")
    ambiguous, _ = _run("volume.increase", status="ambiguous")
    inventory = _snapshot()
    client = _FakeHa()

    def states():
        return [{"entity_id": "media_player.frame", "state": "off", "attributes": {}}]

    client.get_states = states
    HaExecutor(inventory.registry, inventory.bindings, client).execute(
        ResolvedCommand(
            status="resolved",
            intent="device.turn_on",
            semantic_target_id="media_player.frame",
            execution_target_id="media_player.frame",
            arguments={},
            candidates=("media_player.frame",),
        ),
        _Quiet(),
    )
    assert light.calls == []
    assert missing.calls == []
    assert ambiguous.calls == []
    assert client.calls == [("homeassistant", "toggle", {"entity_id": "remote.frame"})]


def test_volume_failure_is_execution_failed():
    client, result = _run("volume.set", "30 процентов", fail=True)
    assert client.calls == []
    assert result.payload["status"] == "execution_failed"
    assert "super-secret-token" not in result.payload["reason"]


def test_volume_phrases_keep_direction_in_the_intent():
    registry = DeviceRegistry.from_static()
    expected = {
        "Поставь громкость телевизора на 30 процентов": ("volume.set", 30),
        "Увеличь громкость телевизора на 10 процентов": ("volume.increase", 10),
        "Уменьши громкость телевизора на 10 процентов": ("volume.decrease", 10),
        "Убавь громкость телевизора на 10 процентов": ("volume.decrease", 10),
        "Сделай телевизор громче": ("volume.increase", None),
        "Сделай телевизор тише": ("volume.decrease", None),
        "Сделай телевизор потише": ("volume.decrease", None),
    }
    for text, (intent, points) in expected.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == intent
        assert parsed.slots.device_type == "tv"
        assert percent_points(parsed.value or "") == points


def test_sound_owns_the_power_verb():
    registry = DeviceRegistry.from_static()
    muted = (
        "Выключи звук на телевизоре",
        "Убери звук на телевизоре",
        "Отключи звук на телевизоре",
    )
    unmuted = (
        "Включи звук на телевизоре",
        "Верни звук на телевизоре",
    )
    for text in muted:
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "volume.mute"
        assert parsed.slots.device_type == "tv"
        assert parsed.value is None
    for text in unmuted:
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "volume.unmute"
        assert parsed.slots.device_type == "tv"
    power = {
        "Включи телевизор": "device.turn_on",
        "Выключи телевизор": "device.turn_off",
        "Вруби телевизор": "device.turn_on",
        "Включи свет": "device.turn_on",
        "Выключи свет": "device.turn_off",
    }
    for text, intent in power.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == intent
        if "свет" in text:
            assert parsed.slots.device_type == "light"
        else:
            assert parsed.slots.device_type == "tv"
    panel = parse_deterministic("Включи звуковая панель", registry)
    assert panel.intent == "device.turn_on"
    assert panel.slots.device_type == "soundbar"


def test_mute_sets_the_desired_audio_state_without_power():
    inventory = _snapshot()
    assert "volume.mute" in inventory.registry.get("media_player.frame").capabilities
    assert "volume.unmute" in inventory.registry.get("media_player.frame").capabilities
    mute, mute_result = _run("volume.mute")
    unmute, _ = _run("volume.unmute")
    assert mute.reads == 0
    assert unmute.reads == 0
    assert mute.calls == [
        ("media_player", "volume_mute", {"entity_id": "media_player.frame", "is_volume_muted": True})
    ]
    assert unmute.calls == [
        ("media_player", "volume_mute", {"entity_id": "media_player.frame", "is_volume_muted": False})
    ]
    assert "remote" not in str(mute.calls)
    assert "toggle" not in str(mute.calls)
    assert mute_result.payload["execution"]["is_volume_muted"] is True
    light, light_result = _run("volume.mute", entity="light.lampa")
    assert light.calls == []
    assert light_result.error == "execution target lacks capability"


def test_absolute_trace_shows_volume_level(caplog):
    caplog.set_level(logging.INFO, logger="ollama_runner.request")
    _client, result = _run("volume.set", "на 30 процентов")
    command = command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(
            intent="volume.set",
            target=Target(device_type="tv", mention="телевизора"),
            arguments={"value": "на 30 процентов"},
        ),
        resolution_status="resolved",
        device_id="media_player.frame",
        resolved_type="tv",
    )
    emit_request(RequestResult(1, "Поставь громкость телевизора на 30 процентов", command.status, (command,)))
    assert "intent=volume.set" in caplog.text
    assert "service=media_player.volume_set" in caplog.text
    assert "entity=media_player.frame" in caplog.text
    assert "volume_level=0.30" in caplog.text
