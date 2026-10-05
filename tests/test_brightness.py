from ollama_runner.ha.execute import HaExecutor, planned_call
from ollama_runner.ha.normalize import Binding, HaExecutionBinding
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import ResolvedCommand
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


def _run(intent: str, entity: str, value: str = "") -> list[tuple]:
    inventory = _snapshot()
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    resolved = ResolvedCommand(
        status="resolved",
        intent=intent,
        semantic_target_id=entity,
        execution_target_id=entity,
        arguments={"value": value} if value else {},
        candidates=(entity,),
    )
    executor.execute(resolved, _Quiet())
    return client.calls


def test_brightness_set_sends_percent():
    calls = _run("brightness.set", "light.bulb_e27_lemon_3", "на 50 процентов")
    assert calls == [("light", "turn_on", {"entity_id": "light.bulb_e27_lemon_3", "brightness_pct": 50})]


def test_brightness_increase_and_decrease_send_signed_steps():
    assert _run("brightness.increase", "light.lampa", "20 процентов") == [
        ("light", "turn_on", {"entity_id": "light.lampa", "brightness_step_pct": 20})
    ]
    assert _run("brightness.decrease", "light.lampa", "20") == [
        ("light", "turn_on", {"entity_id": "light.lampa", "brightness_step_pct": -20})
    ]


def test_relative_without_a_value_uses_the_explicit_default():
    assert _run("brightness.increase", "light.lampa") == [
        ("light", "turn_on", {"entity_id": "light.lampa", "brightness_step_pct": 10})
    ]
    assert _run("brightness.decrease", "light.lampa") == [
        ("light", "turn_on", {"entity_id": "light.lampa", "brightness_step_pct": -10})
    ]


def test_non_numeric_magnitude_and_out_of_range_are_not_sent():
    assert _run("brightness.increase", "light.lampa", "немного") == []
    assert _run("brightness.set", "light.lampa", "150 процентов") == []
    assert _run("brightness.set", "light.lampa", "") == []
    spoken = planned_call(
        "brightness.set",
        Binding(HaExecutionBinding("light.lampa", "light")),
        "на пятьдесят процентов",
    )
    assert spoken is not None
    assert spoken.brightness_pct == 50


def test_brightness_on_unsupported_device_does_not_call():
    assert _run("brightness.set", "media_player.frame", "50 процентов") == []


def test_unresolved_brightness_does_not_call():
    inventory = _snapshot()
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    executor.execute(
        ResolvedCommand(
            status="not_found",
            intent="brightness.set",
            semantic_target_id=None,
            execution_target_id=None,
            arguments={"value": "50 процентов"},
            candidates=(),
        ),
        _Quiet(),
    )
    assert client.calls == []


def test_numbered_lamps_keep_their_targets():
    inventory = _snapshot()
    resolver = CapabilityResolver(inventory.registry)
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, inventory.bindings, client, perform=False)
    expected = {
        "Поставь яркость лампы 1 на 30 процентов": ("light.bulb_e27_lemon_3", 30, None),
        "Поставь яркость лампы один на 30 процентов": ("light.bulb_e27_lemon_3", 30, None),
        "Поставь яркость лампы 2 на 30 процентов": ("light.bulb_e27_lemon_3_2", 30, None),
        "Сделай лампу 2 потемнее": ("light.bulb_e27_lemon_3_2", None, -10),
    }
    for text, (entity, pct, step) in expected.items():
        parsed = parse_deterministic(text, inventory.registry)
        assert parsed.handled, text
        resolved = resolver.resolve(parsed.command(), text=text)
        assert resolved.status == "resolved", text
        assert resolved.execution_target_id == entity, text
        executor.planned.clear()
        executor.execute(resolved, _Quiet())
        call = executor.planned[-1]
        assert call.entity_id == entity
        assert call.service == "turn_on"
        assert call.brightness_pct == pct
        assert call.brightness_step_pct == step
    assert client.calls == []
