from ollama_runner.ha.execute import HaExecutor, planned_call
from ollama_runner.ha.normalize import Binding, HaExecutionBinding
from ollama_runner.ha.route import ExecutionRouter
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.nlu.values import percent_points
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import ResolvedCommand
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _area, _snapshot, build_inventory


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


def test_directional_brightness_keeps_a_positive_magnitude():
    registry = DeviceRegistry.from_static()
    decrease = {
        "уменьши яркость света на 10 процентов": 10,
        "убавь яркость света на 10 процентов": 10,
        "снизь яркость света на 10 процентов": 10,
        "сделай свет темнее на 10 процентов": 10,
    }
    increase = {
        "увеличь яркость света на 10 процентов": 10,
        "повысь яркость света на 10 процентов": 10,
        "сделай свет ярче на 10 процентов": 10,
    }
    absolute = {
        "поставь яркость света на 10 процентов": 10,
        "установи яркость света на 30 процентов": 30,
        "сделай яркость света 50 процентов": 50,
    }
    for text, points in decrease.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "brightness.decrease", text
        assert percent_points(parsed.value) == points, text
        assert parsed.value is None or not str(parsed.value).startswith("-"), text
    for text, points in increase.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "brightness.increase", text
        assert percent_points(parsed.value) == points, text
    for text, points in absolute.items():
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "brightness.set", text
        assert percent_points(parsed.value) == points, text

    named = parse_deterministic("уменьши яркость света на 10 процентов", registry)
    assert named.slots.device_type == "light"
    assert named.mention == "света"
    bare = parse_deterministic("уменьши яркость на 10 процентов", registry)
    assert bare.intent == "brightness.decrease"
    assert percent_points(bare.value) == 10
    for text in ("уменьши яркость света", "убавь яркость света", "сделай свет темнее"):
        parsed = parse_deterministic(text, registry)
        assert parsed.intent == "brightness.decrease", text
        assert parsed.value is None, text


def _one_light():
    return build_inventory(
        areas=[_area("room", "Room")],
        devices=[],
        entities=[{
            "entity_id": "light.only",
            "device_id": None,
            "area_id": "room",
            "platform": "group",
            "disabled_by": None,
            "hidden_by": None,
            "entity_category": None,
            "labels": [],
            "name": None,
            "original_name": "Свет",
            "aliases": [],
        }],
        labels=[],
        states=[{
            "entity_id": "light.only",
            "state": "on",
            "attributes": {"friendly_name": "Свет", "supported_color_modes": ["hs"]},
        }],
        services=[{"domain": "light", "services": {"turn_on": {}, "turn_off": {}}}],
        entity_details={"light.only": {"aliases": [], "capabilities": {"supported_color_modes": ["hs"]}}},
    )


def test_relative_decrease_sends_a_negative_step_twice():
    inventory = _one_light()
    client = _FakeHa()
    resolver = CapabilityResolver(inventory.registry)
    router = ExecutionRouter(inventory.registry, inventory.bindings, client)
    text = "уменьши яркость света на 10 процентов"
    for _ in range(2):
        parsed = parse_deterministic(text, inventory.registry)
        assert parsed.intent == "brightness.decrease"
        resolved = resolver.resolve(parsed.command(), text=text)
        assert resolved.status == "resolved"
        router.execute(resolved, _Quiet())
    assert client.calls == [
        ("light", "turn_on", {"entity_id": "light.only", "brightness_step_pct": -10}),
        ("light", "turn_on", {"entity_id": "light.only", "brightness_step_pct": -10}),
    ]
    assert all("brightness_pct" not in payload for _, _, payload in client.calls)


def test_unnamed_decrease_stays_ambiguous():
    inventory = _snapshot()
    client = _FakeHa()
    resolver = CapabilityResolver(inventory.registry)
    router = ExecutionRouter(inventory.registry, inventory.bindings, client)
    text = "уменьши яркость на 10 процентов"
    parsed = parse_deterministic(text, inventory.registry)
    assert parsed.intent == "brightness.decrease"
    resolved = resolver.resolve(parsed.command(), text=text)
    assert resolved.status == "ambiguous"
    router.execute(resolved, _Quiet())
    assert client.calls == []
