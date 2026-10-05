from ollama_runner.ha.execute import HaExecutor
from ollama_runner.ha.normalize import build_inventory
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import NotCommandOutcome
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _area


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def call_service(self, domain: str, service: str, data: dict) -> int:
        self.calls.append((domain, service, data))
        return 200


class _NoModel:
    def parse(self, text: str, context) -> NotCommandOutcome:
        raise AssertionError(text)


def _light(entity_id: str, *, area: str, name: str, members: list[str] | None = None, state: str = "on") -> tuple[dict, dict]:
    entity = {
        "entity_id": entity_id,
        "device_id": None,
        "area_id": area,
        "platform": "group" if members else "tuya_local",
        "disabled_by": None,
        "hidden_by": None,
        "entity_category": None,
        "labels": [],
        "name": None,
        "original_name": name,
        "aliases": [],
    }
    attributes = {"friendly_name": name, "supported_color_modes": ["hs"]}
    if members:
        attributes["entity_id"] = members
    return entity, {
        "entity_id": entity_id,
        "state": state,
        "attributes": attributes,
    }


def _media(entity_id: str, *, state: str) -> tuple[dict, dict]:
    entity = {
        "entity_id": entity_id,
        "device_id": None,
        "area_id": "room",
        "platform": "samsungtv",
        "disabled_by": None,
        "hidden_by": None,
        "entity_category": None,
        "labels": [],
        "name": None,
        "original_name": entity_id,
        "aliases": [],
    }
    return entity, {
        "entity_id": entity_id,
        "state": state,
        "attributes": {
            "friendly_name": entity_id,
            "device_class": "tv",
            "supported_features": 16385,
        },
    }


def _inventory(lights: list[tuple[dict, dict]], media: list[tuple[dict, dict]] | None = None):
    areas = [_area("room", "Room", ["комната"]), _area("kitchen", "Kitchen", ["кухня"]), _area("bedroom", "Bedroom", ["спальня"])]
    entities = []
    states = []
    for entity, state in lights + (media or []):
        entities.append(entity)
        states.append(state)
    return build_inventory(
        areas=areas,
        devices=[],
        entities=entities,
        labels=[],
        states=states,
        services=[
            {"domain": "light", "services": {"turn_on": {}, "turn_off": {}}},
            {"domain": "media_player", "services": {"media_pause": {}, "media_play": {}}},
        ],
        entity_details={entity["entity_id"]: {"aliases": []} for entity, _ in lights + (media or [])},
    )


def _run(inventory, text: str):
    client = _FakeHa()
    members = {
        entity_id: binding.members
        for entity_id, binding in inventory.bindings.items()
        if binding.members
    }
    pipeline = SemanticPipeline(
        _NoModel(),
        CapabilityResolver(inventory.registry, members=members),
        HaExecutor(inventory.registry, inventory.bindings, client),
        inventory.registry,
    )
    result = pipeline.run(text, _Quiet(), state=inventory.state)
    return client.calls, result, pipeline.last_request


def _room_and_desk(*, room: str = "on", desk: str = "off"):
    room_entity, room_state = _light("light.room", area="room", name="Комната", members=["light.a", "light.b"], state=room)
    a_entity, a_state = _light("light.a", area="room", name="A", state="on")
    b_entity, b_state = _light("light.b", area="room", name="Лампа 2", state="on")
    desk_entity, desk_state = _light("light.desk", area="room", name="Стол", state=desk)
    return _inventory([(room_entity, room_state), (a_entity, a_state), (b_entity, b_state), (desk_entity, desk_state)])


def test_group_covers_its_members():
    inventory = _inventory([
        _light("light.room", area="room", name="Комната", members=["light.a", "light.b"]),
        _light("light.a", area="room", name="A"),
        _light("light.b", area="room", name="B"),
    ])
    calls, _, _ = _run(inventory, "сделай темнее")
    assert calls == [("light", "turn_on", {"entity_id": "light.room", "brightness_step_pct": -20})]


def test_independent_light_is_called_with_its_group():
    calls, _, _ = _run(_room_and_desk(), "сделай темнее")
    assert calls == [
        ("light", "turn_on", {"entity_id": "light.desk", "brightness_step_pct": -20}),
        ("light", "turn_on", {"entity_id": "light.room", "brightness_step_pct": -20}),
    ]


def test_nested_group_keeps_only_the_outer_group():
    inventory = _inventory([
        _light("light.home", area="room", name="Дом", members=["light.room", "light.desk"]),
        _light("light.room", area="room", name="Комната", members=["light.a", "light.b"]),
        _light("light.a", area="room", name="A"),
        _light("light.b", area="room", name="B"),
        _light("light.desk", area="room", name="Стол"),
    ])
    calls, _, _ = _run(inventory, "сделай темнее")
    assert calls == [("light", "turn_on", {"entity_id": "light.home", "brightness_step_pct": -20})]


def test_two_independent_lights_are_both_called():
    inventory = _inventory([
        _light("light.desk", area="room", name="Стол"),
        _light("light.lamp", area="room", name="Лампа"),
    ])
    calls, _, _ = _run(inventory, "сделай темнее")
    assert calls == [
        ("light", "turn_on", {"entity_id": "light.desk", "brightness_step_pct": -20}),
        ("light", "turn_on", {"entity_id": "light.lamp", "brightness_step_pct": -20}),
    ]


def test_explicit_member_is_not_replaced_by_its_group():
    calls, _, _ = _run(_room_and_desk(), "сделай лампу 2 темнее")
    assert calls == [("light", "turn_on", {"entity_id": "light.b", "brightness_step_pct": -20})]


def test_explicit_area_limits_maximal_targets():
    inventory = _inventory([
        _light("light.kitchen", area="kitchen", name="Кухня", members=["light.k1"]),
        _light("light.k1", area="kitchen", name="Кухня 1"),
        _light("light.bed", area="bedroom", name="Спальня"),
    ])
    calls, _, _ = _run(inventory, "сделай свет на кухне темнее")
    assert calls == [("light", "turn_on", {"entity_id": "light.kitchen", "brightness_step_pct": -20})]


def test_brightness_steps_and_implicit_phrases():
    inventory = _room_and_desk()
    darker, _, _ = _run(inventory, "сделай свет темнее")
    brighter, _, _ = _run(inventory, "сделай свет ярче")
    exact_down, _, _ = _run(inventory, "сделай свет темнее на 10 процентов")
    exact_up, _, _ = _run(inventory, "сделай свет ярче на 10 процентов")
    too_dark, _, request = _run(inventory, "слишком темно")
    too_bright, _, _ = _run(inventory, "слишком ярко")
    absolute, _, _ = _run(inventory, "поставь яркость на 40 процентов")
    assert {call[2]["brightness_step_pct"] for call in darker} == {-20}
    assert {call[2]["brightness_step_pct"] for call in brighter} == {20}
    assert {call[2]["brightness_step_pct"] for call in exact_down} == {-10}
    assert {call[2]["brightness_step_pct"] for call in exact_up} == {10}
    assert request is not None
    assert request.commands[0].intent == "brightness.increase"
    assert {call[2]["brightness_step_pct"] for call in too_dark} == {20}
    assert {call[2]["brightness_step_pct"] for call in too_bright} == {-20}
    assert {call[2]["brightness_pct"] for call in absolute} == {40}
    for calls in (darker, brighter, exact_down, exact_up, too_dark, too_bright, absolute):
        assert {call[2]["entity_id"] for call in calls} == {"light.desk", "light.room"}


def test_light_shorthand_toggles_each_maximal_target():
    off, _, _ = _run(_inventory([_light("light.only", area="room", name="Свет", state="off")]), "свет")
    on, _, _ = _run(_inventory([_light("light.only", area="room", name="Свет", state="on")]), "свет")
    group_on, _, _ = _run(_inventory([
        _light("light.room", area="room", name="Комната", members=["light.a", "light.b"], state="on"),
        _light("light.a", area="room", name="A", state="on"),
        _light("light.b", area="room", name="B", state="off"),
    ]), "свет")
    group_off, _, _ = _run(_inventory([
        _light("light.room", area="room", name="Комната", members=["light.a", "light.b"], state="off"),
        _light("light.a", area="room", name="A", state="on"),
        _light("light.b", area="room", name="B", state="on"),
    ]), "свет")
    mixed, _, _ = _run(_room_and_desk(room="on", desk="off"), "свет")
    assert off == [("light", "turn_on", {"entity_id": "light.only"})]
    assert on == [("light", "turn_off", {"entity_id": "light.only"})]
    assert group_on == [("light", "turn_off", {"entity_id": "light.room"})]
    assert group_off == [("light", "turn_on", {"entity_id": "light.room"})]
    assert mixed == [
        ("light", "turn_on", {"entity_id": "light.desk"}),
        ("light", "turn_off", {"entity_id": "light.room"}),
    ]


def test_sound_follows_the_active_media_session():
    playing = _inventory([], [_media("media_player.playing", state="playing"), _media("media_player.idle", state="on")])
    paused = _inventory([], [_media("media_player.paused", state="paused")])
    both = _inventory([], [_media("media_player.one", state="playing"), _media("media_player.two", state="paused")])
    quiet = _inventory([], [_media("media_player.off", state="off"), _media("media_player.idle", state="idle")])
    play_calls, _, play_request = _run(playing, "звук")
    pause_calls, _, pause_request = _run(paused, "звук")
    both_calls, both_result, _ = _run(both, "звук")
    quiet_calls, quiet_result, _ = _run(quiet, "звук")
    parsed = parse_deterministic("звук", DeviceRegistry.from_static())
    assert parsed.intent == "media.toggle"
    assert play_calls == [("media_player", "media_pause", {"entity_id": "media_player.playing"})]
    assert play_request is not None and play_request.commands[0].intent == "media.pause"
    assert pause_calls == [("media_player", "media_play", {"entity_id": "media_player.paused"})]
    assert pause_request is not None and pause_request.commands[0].intent == "media.resume"
    assert both_calls == []
    assert both_result.payload["status"] == "ambiguous"
    assert quiet_calls == []
    assert quiet_result.payload["status"] == "clarify"
    assert quiet_result.payload["reason"] == "no_active_device"


def test_named_power_and_brightness_phrases_keep_their_intents():
    registry = DeviceRegistry.from_static()
    power = parse_deterministic("включи свет", registry)
    off = parse_deterministic("выключи свет", registry)
    absolute = parse_deterministic("поставь яркость света на 10 процентов", registry)
    down = parse_deterministic("уменьши яркость света на 10 процентов", registry)
    up = parse_deterministic("увеличь яркость света на 10 процентов", registry)
    assert power.intent == "device.turn_on"
    assert off.intent == "device.turn_off"
    assert absolute.intent == "brightness.set"
    assert down.intent == "brightness.decrease"
    assert up.intent == "brightness.increase"
    resolved = CapabilityResolver(registry).resolve(power.command(), text="включи свет")
    assert resolved.status == "resolved"
    assert resolved.execution_target_id == "light_common"
    assert resolved.execution_target_ids == ()
