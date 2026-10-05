import logging

from ollama_runner.ha.execute import DEFAULT_COLOR_TEMP_STEP_K, HaExecutor
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import ResolvedCommand
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _snapshot


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self, states: list | None = None) -> None:
        self.calls: list[tuple] = []
        self.states = [] if states is None else states
        self.state_reads = 0

    def call_service(self, domain: str, service: str, data: dict) -> int:
        self.calls.append((domain, service, data))
        return 200

    def get_states(self) -> list:
        self.state_reads += 1
        return self.states


def _kelvin(entity: str, value: int) -> dict:
    return {
        "entity_id": entity,
        "state": "on",
        "attributes": {"color_temp_kelvin": value},
    }


def _execute(intent: str, entity: str, value: str = "", states: list | None = None):
    inventory = _snapshot()
    client = _FakeHa(states)
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    resolved = ResolvedCommand(
        status="resolved",
        intent=intent,
        semantic_target_id=entity,
        execution_target_id=entity,
        arguments={"value": value} if value else {},
        candidates=(entity,),
    )
    result = executor.execute(resolved, _Quiet())
    return client, result


def test_phrases_become_color_temperature_commands():
    registry = DeviceRegistry.from_static()
    warmer = parse_deterministic("Сделай свет теплее", registry)
    cooler = parse_deterministic("Сделай лампу один холоднее", registry)
    absolute = parse_deterministic("Поставь температуру света 4000 кельвинов", registry)
    lamp = parse_deterministic("Поставь лампу 1 на 3500 кельвинов", registry)
    named = parse_deterministic("Сделай свет теплый белый", registry)
    assert warmer.intent == "color_temperature.warmer"
    assert warmer.value is None
    assert warmer.slots.device_type == "light"
    assert cooler.intent == "color_temperature.cooler"
    assert cooler.slots.ordinal == 1
    assert absolute.intent == "color_temperature.set"
    assert absolute.value == "4000"
    assert lamp.intent == "color_temperature.set"
    assert lamp.value == "3500"
    assert lamp.slots.ordinal == 1
    assert named.intent == "color.set"
    assert named.value == "теплый белый"


def test_absolute_kelvin_uses_the_service_parameter():
    client, result = _execute("color_temperature.set", "light.lampa", "4000")
    assert client.state_reads == 0
    assert client.calls == [("light", "turn_on", {"entity_id": "light.lampa", "color_temp_kelvin": 4000})]
    assert result.payload["execution"]["color_temp_kelvin"] == 4000


def test_absolute_outside_device_range_is_not_sent():
    below, below_result = _execute("color_temperature.set", "light.lampa", "2000")
    above, above_result = _execute("color_temperature.set", "light.lampa", "7000")
    assert below.calls == []
    assert above.calls == []
    assert below_result.payload["status"] == "unsupported"
    assert below_result.payload["reason"] == "color_temp_out_of_range"
    assert above_result.payload["reason"] == "color_temp_out_of_range"


def test_fractional_reported_kelvin_is_rounded_before_the_step():
    client, result = _execute(
        "color_temperature.warmer",
        "light.lampa",
        states=[_kelvin("light.lampa", 3999.6)],
    )
    assert client.calls == [("light", "turn_on", {"entity_id": "light.lampa", "color_temp_kelvin": 3600})]
    assert result.payload["execution"]["current_kelvin"] == 4000


def test_warmer_decreases_kelvin_and_cooler_increases_it():
    warmer, _ = _execute(
        "color_temperature.warmer",
        "light.bulb_e27_lemon_3",
        states=[_kelvin("light.bulb_e27_lemon_3", 4000)],
    )
    cooler, _ = _execute(
        "color_temperature.cooler",
        "light.bulb_e27_lemon_3",
        states=[_kelvin("light.bulb_e27_lemon_3", 4000)],
    )
    assert warmer.calls == [
        ("light", "turn_on", {"entity_id": "light.bulb_e27_lemon_3", "color_temp_kelvin": 4000 - DEFAULT_COLOR_TEMP_STEP_K})
    ]
    assert cooler.calls == [
        ("light", "turn_on", {"entity_id": "light.bulb_e27_lemon_3", "color_temp_kelvin": 4000 + DEFAULT_COLOR_TEMP_STEP_K})
    ]


def test_relative_clamps_to_the_advertised_range():
    low, _ = _execute(
        "color_temperature.warmer",
        "light.lampa",
        states=[_kelvin("light.lampa", 2900)],
    )
    high, _ = _execute(
        "color_temperature.cooler",
        "light.lampa",
        states=[_kelvin("light.lampa", 6300)],
    )
    assert low.calls == [("light", "turn_on", {"entity_id": "light.lampa", "color_temp_kelvin": 2700})]
    assert high.calls == [("light", "turn_on", {"entity_id": "light.lampa", "color_temp_kelvin": 6500})]


def test_explicit_invalid_is_rejected_while_relative_clamps():
    invalid, _ = _execute("color_temperature.set", "light.lampa", "2300")
    relative, _ = _execute(
        "color_temperature.warmer",
        "light.lampa",
        states=[_kelvin("light.lampa", 2700)],
    )
    assert invalid.calls == []
    assert relative.calls == [("light", "turn_on", {"entity_id": "light.lampa", "color_temp_kelvin": 2700})]


def test_group_relative_without_current_kelvin_is_not_sent():
    client, result = _execute(
        "color_temperature.warmer",
        "light.lampa",
        states=[{"entity_id": "light.lampa", "state": "on", "attributes": {}}],
    )
    assert client.calls == []
    assert result.payload["status"] == "unsupported"
    assert result.payload["reason"] == "color_temp_unavailable"


def test_group_absolute_still_sends_when_limits_exist():
    client, _ = _execute("color_temperature.set", "light.lampa", "3500")
    assert client.calls == [("light", "turn_on", {"entity_id": "light.lampa", "color_temp_kelvin": 3500})]


def test_unsupported_light_tv_and_not_found_do_not_call():
    light, _ = _execute("color_temperature.set", "light.bulb_e27_lemon_3_2", "4000")
    tv, _ = _execute("color_temperature.cooler", "media_player.frame")
    inventory = _snapshot()
    missing = _FakeHa()
    HaExecutor(inventory.registry, inventory.bindings, missing).execute(
        ResolvedCommand(
            status="not_found",
            intent="color_temperature.set",
            semantic_target_id=None,
            execution_target_id=None,
            arguments={"value": "4000"},
            candidates=(),
        ),
        _Quiet(),
    )
    assert light.calls == []
    assert tv.calls == []
    assert missing.calls == []


def test_relative_calculation_is_in_the_debug_trace(caplog):
    caplog.set_level(logging.DEBUG, logger="ollama_runner.request")
    inventory = _snapshot()
    client = _FakeHa([_kelvin("light.bulb_e27_lemon_3", 4000)])
    pipeline = SemanticPipeline(
        None,
        CapabilityResolver(inventory.registry),
        HaExecutor(inventory.registry, inventory.bindings, client),
        inventory.registry,
    )
    pipeline.run("Сделай лампу 1 теплее", _Quiet(), state=inventory.state)
    request = pipeline.last_request
    assert request is not None
    command = request.commands[0]
    assert command.intent == "color_temperature.warmer"
    assert command.entity_id == "light.bulb_e27_lemon_3"
    assert command.color_temp_kelvin == 4000 - DEFAULT_COLOR_TEMP_STEP_K
    assert "current_kelvin=4000" in caplog.text
    assert "direction=warmer" in caplog.text
    assert f"step_kelvin={DEFAULT_COLOR_TEMP_STEP_K}" in caplog.text
    assert f"target_kelvin={4000 - DEFAULT_COLOR_TEMP_STEP_K}" in caplog.text
    assert "color_temp_kelvin=" in caplog.text
