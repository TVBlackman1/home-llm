import logging

from ollama_runner.ha.execute import HaExecutor
from ollama_runner.ha.normalize import Binding
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.request import RequestResult, command_from_result, emit_request
from ollama_runner.semantic import ResolvedCommand, SemanticCommand
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


def _execute(entity: str, value: str, *, modes: frozenset[str] | None = None, status: str = "resolved"):
    inventory = _snapshot()
    bindings = dict(inventory.bindings)
    if modes is not None:
        current = bindings[entity]
        bindings[entity] = Binding(
            current.default,
            dict(current.capability_bindings),
            current.members,
            current.min_color_temp_kelvin,
            current.max_color_temp_kelvin,
            modes,
        )
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, bindings, client)
    result = executor.execute(
        ResolvedCommand(
            status=status,
            intent="color.set",
            semantic_target_id=entity,
            execution_target_id=entity if status == "resolved" else None,
            arguments={"value": value},
            candidates=(entity,) if status == "resolved" else (),
        ),
        _Quiet(),
    )
    return client, result


def test_supported_phrases_keep_color_set():
    registry = DeviceRegistry.from_static()
    red = parse_deterministic("Сделай свет красным", registry)
    blue = parse_deterministic("Сделай свет синим", registry)
    green = parse_deterministic("Сделай свет зеленым", registry)
    warm = parse_deterministic("Сделай свет теплый белый", registry)
    warmer = parse_deterministic("Сделай свет теплее", registry)
    assert (red.intent, red.value, red.mention) == ("color.set", "красный", "свет")
    assert (blue.intent, blue.value) == ("color.set", "синий")
    assert (green.intent, green.value) == ("color.set", "зеленый")
    assert (warm.intent, warm.value) == ("color.set", "теплый белый")
    assert warmer.intent == "color_temperature.warmer"


def test_inflected_lamp_and_white_phrases_are_not_commands():
    registry = DeviceRegistry.from_static()
    for text in (
        "Сделай лампу 1 красной",
        "Сделай лампу один синей",
        "Сделай лампу 2 зеленой",
        "Сделай свет теплым белым",
        "Сделай свет холодным белым",
        "Сделай свет белым",
    ):
        parsed = parse_deterministic(text, registry)
        assert parsed.handled is False
        assert parsed.intent is None


def test_chromatic_colors_send_rgb_without_brightness_or_kelvin():
    red, _ = _execute("light.lampa", "красный")
    blue, _ = _execute("light.bulb_e27_lemon_3", "синий")
    green, _ = _execute("light.bulb_e27_lemon_3_2", "зеленый")
    assert red.calls == [("light", "turn_on", {"entity_id": "light.lampa", "rgb_color": [255, 0, 0]})]
    assert blue.calls == [
        ("light", "turn_on", {"entity_id": "light.bulb_e27_lemon_3", "rgb_color": [0, 0, 255]})
    ]
    assert green.calls == [
        ("light", "turn_on", {"entity_id": "light.bulb_e27_lemon_3_2", "rgb_color": [0, 255, 0]})
    ]
    for calls in (red.calls, blue.calls, green.calls):
        payload = calls[0][2]
        assert "brightness_pct" not in payload
        assert "brightness" not in payload
        assert "color_temp_kelvin" not in payload
        assert "hs_color" not in payload


def test_warm_white_stays_a_named_color():
    client, result = _execute("light.lampa", "теплый белый")
    assert client.calls == [
        ("light", "turn_on", {"entity_id": "light.lampa", "rgb_color": [255, 230, 204]})
    ]
    assert "color_temp_kelvin" not in client.calls[0][2]
    assert result.payload["execution"]["rgb_color"] == [255, 230, 204]


def test_plain_white_and_cool_white_are_not_sent():
    for name in ("белый", "холодный белый"):
        client, result = _execute("light.lampa", name)
        assert client.calls == []
        assert result.payload["reason"] == "color_unsupported"


def test_color_without_a_chromatic_mode_tv_and_misses_are_not_sent():
    temperature_only, _ = _execute("light.lampa", "красный", modes=frozenset({"color_temp"}))
    missing, _ = _execute("light.lampa", "красный", status="not_found")
    ambiguous, _ = _execute("light.lampa", "красный", status="ambiguous")
    inventory = _snapshot()
    tv = _FakeHa()
    HaExecutor(inventory.registry, inventory.bindings, tv).execute(
        ResolvedCommand(
            status="resolved",
            intent="color.set",
            semantic_target_id="media_player.frame",
            execution_target_id="media_player.frame",
            arguments={"value": "красный"},
            candidates=("media_player.frame",),
        ),
        _Quiet(),
    )
    assert temperature_only.calls == []
    assert missing.calls == []
    assert ambiguous.calls == []
    assert tv.calls == []


def test_execute_trace_shows_rgb(caplog):
    caplog.set_level(logging.INFO, logger="ollama_runner.request")
    _client, result = _execute("light.lampa", "красный")
    command = command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(intent="color.set", arguments={"value": "красный"}),
        resolution_status="resolved",
        device_id="light.lampa",
    )
    emit_request(RequestResult(1, "Сделай свет красным", command.status, (command,)))
    assert "intent=color.set" in caplog.text
    assert "value=красный" in caplog.text
    assert "service=light.turn_on" in caplog.text
    assert "entity=light.lampa" in caplog.text
    assert "rgb_color=[255, 0, 0]" in caplog.text
    assert "brightness" not in caplog.text.split("rgb_color")[-1]
