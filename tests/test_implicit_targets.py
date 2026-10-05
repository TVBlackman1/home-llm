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


def _light(
    entity_id: str,
    *,
    area: str,
    name: str,
    members: list[str] | None = None,
    state: str = "on",
    labels: list[str] | None = None,
    aliases: list[str] | None = None,
) -> tuple[dict, dict]:
    entity = {
        "entity_id": entity_id,
        "device_id": None,
        "area_id": area,
        "platform": "group" if members else "tuya_local",
        "disabled_by": None,
        "hidden_by": None,
        "entity_category": None,
        "labels": labels or [],
        "name": None,
        "original_name": name,
        "aliases": aliases or [],
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


def _inventory(
    lights: list[tuple[dict, dict]],
    media: list[tuple[dict, dict]] | None = None,
    labels: list[dict] | None = None,
):
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
        labels=labels or [],
        states=states,
        services=[
            {"domain": "light", "services": {"turn_on": {}, "turn_off": {}}},
            {"domain": "media_player", "services": {"media_pause": {}, "media_play": {}}},
        ],
        entity_details={
            entity["entity_id"]: {"aliases": entity.get("aliases") or []}
            for entity, _ in lights + (media or [])
        },
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
    assert "light_common" in resolved.execution_target_ids
    assert "light_common_kitchen" in resolved.execution_target_ids
    assert len(resolved.execution_target_ids) > 1


def _ids(calls) -> set[str]:
    return {payload["entity_id"] for _, _, payload in calls}


def test_area_boundary_excludes_the_other_room_and_its_members():
    inventory = _inventory([
        _light("light.kitchen_group", area="kitchen", name="Кухня", members=["light.kitchen_a", "light.kitchen_b"]),
        _light("light.kitchen_a", area="kitchen", name="Кухня A"),
        _light("light.kitchen_b", area="kitchen", name="Кухня B"),
        _light("light.bedroom_group", area="bedroom", name="Спальня", members=["light.bedroom_a", "light.bedroom_b"]),
        _light("light.bedroom_a", area="bedroom", name="Спальня A"),
        _light("light.bedroom_b", area="bedroom", name="Спальня B"),
    ])
    kitchen, _, _ = _run(inventory, "сделай свет на кухне темнее")
    bedroom, _, _ = _run(inventory, "выключи свет в спальне")
    assert _ids(kitchen) == {"light.kitchen_group"}
    assert "light.kitchen_a" not in _ids(kitchen)
    assert "light.kitchen_b" not in _ids(kitchen)
    assert "light.bedroom_group" not in _ids(kitchen)
    assert "light.bedroom_a" not in _ids(kitchen)
    assert "light.bedroom_b" not in _ids(kitchen)
    assert _ids(bedroom) == {"light.bedroom_group"}
    assert "light.kitchen_group" not in _ids(bedroom)
    assert "light.kitchen_a" not in _ids(bedroom)
    assert "light.bedroom_a" not in _ids(bedroom)


def test_outside_group_does_not_absorb_an_in_area_member():
    """An explicit area keeps only entities whose own area matches.

    A group outside that area is not a candidate, so it cannot cover an
    in-area member. Containment never crosses the area boundary.
    """

    inventory = _inventory([
        _light("light.bedroom_group", area="bedroom", name="Спальня", members=["light.kitchen_lamp"]),
        _light("light.kitchen_lamp", area="kitchen", name="Кухня"),
        _light("light.bedroom_only", area="bedroom", name="Бра"),
    ])
    calls, _, _ = _run(inventory, "сделай свет на кухне темнее")
    assert _ids(calls) == {"light.kitchen_lamp"}
    assert "light.bedroom_group" not in _ids(calls)
    assert "light.bedroom_only" not in _ids(calls)


def test_in_area_group_is_called_without_its_outside_member():
    inventory = _inventory([
        _light("light.kitchen_group", area="kitchen", name="Кухня", members=["light.bedroom_lamp"]),
        _light("light.bedroom_lamp", area="bedroom", name="Спальня"),
    ])
    calls, _, _ = _run(inventory, "включи свет на кухне")
    assert calls == [("light", "turn_on", {"entity_id": "light.kitchen_group"})]


def test_owner_boundary_excludes_other_owners():
    labels = [
        {"label_id": "label-mama", "name": "owner:mama"},
        {"label_id": "label-masha", "name": "owner:masha"},
    ]
    inventory = _inventory([
        _light("light.mama_group", area="room", name="Мама", members=["light.mama_bulb"], labels=["label-mama"]),
        _light("light.mama_bulb", area="room", name="Мама лампа", labels=["label-mama"]),
        _light("light.masha", area="room", name="Маша", labels=["label-masha"]),
    ], labels=labels)
    calls, _, _ = _run(inventory, "убавь у мамы яркость на 10 процентов")
    assert _ids(calls) == {"light.mama_group"}
    assert "light.mama_bulb" not in _ids(calls)
    assert "light.masha" not in _ids(calls)


def test_unique_alias_does_not_fan_out():
    inventory = _inventory([
        _light("light.room", area="room", name="Комната", members=["light.night", "light.bulb"]),
        _light("light.night", area="room", name="Ночник"),
        _light("light.bulb", area="room", name="Лампа"),
    ])
    calls, _, _ = _run(inventory, "сделай ночник темнее")
    assert calls == [("light", "turn_on", {"entity_id": "light.night", "brightness_step_pct": -20})]


def test_type_only_power_calls_every_maximal_light():
    calls, result, request = _run(_room_and_desk(), "включи свет")
    assert _ids(calls) == {"light.desk", "light.room"}
    assert result.ok is True
    assert request is not None and request.ok is True
    off, off_result, off_request = _run(_room_and_desk(), "выключи свет")
    assert _ids(off) == {"light.desk", "light.room"}
    assert off_result.ok is True
    assert off_request is not None and off_request.ok is True


def test_payload_shape_separates_absolute_and_relative_brightness():
    inventory = _inventory([_light("light.only", area="room", name="Свет")])
    absolute, _, _ = _run(inventory, "поставь яркость света на 10 процентов")
    up, _, _ = _run(inventory, "увеличь яркость света на 10 процентов")
    down, _, _ = _run(inventory, "уменьши яркость света на 10 процентов")
    default_up, _, _ = _run(inventory, "сделай свет ярче")
    default_down, _, _ = _run(inventory, "сделай свет темнее")
    assert absolute == [("light", "turn_on", {"entity_id": "light.only", "brightness_pct": 10})]
    assert up == [("light", "turn_on", {"entity_id": "light.only", "brightness_step_pct": 10})]
    assert down == [("light", "turn_on", {"entity_id": "light.only", "brightness_step_pct": -10})]
    assert default_up == [("light", "turn_on", {"entity_id": "light.only", "brightness_step_pct": 20})]
    assert default_down == [("light", "turn_on", {"entity_id": "light.only", "brightness_step_pct": -20})]
    for calls in (absolute, up, down, default_up, default_down):
        payload = calls[0][2]
        assert ("brightness_pct" in payload) != ("brightness_step_pct" in payload)


def test_repeated_default_decrease_stays_a_step():
    inventory = _inventory([_light("light.only", area="room", name="Свет")])
    first, _, _ = _run(inventory, "сделай свет темнее")
    second, _, _ = _run(inventory, "сделай свет темнее")
    assert first == second == [("light", "turn_on", {"entity_id": "light.only", "brightness_step_pct": -20})]


def test_unknown_light_state_is_not_a_successful_toggle():
    inventory = _inventory([
        _light("light.known", area="room", name="Известный", state="off"),
        _light("light.unknown", area="room", name="Неизвестный", state="unavailable"),
    ])
    calls, result, request = _run(inventory, "свет")
    assert calls == [("light", "turn_on", {"entity_id": "light.known"})]
    assert result.ok is False
    assert request is not None
    assert request.ok is False
    statuses = {command.device_id: command.status for command in request.commands}
    assert statuses["light.known"] == "success"
    assert statuses["light.unknown"] != "success"


class _SecondFails:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def call_service(self, domain: str, service: str, data: dict) -> int:
        self.calls.append((domain, service, data))
        if data["entity_id"] == "light.room":
            raise RuntimeError("rejected")
        return 200


def test_fanout_records_a_rejected_target_without_claiming_success():
    inventory = _room_and_desk()
    client = _SecondFails()
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
    result = pipeline.run("сделай темнее", _Quiet(), state=inventory.state)
    request = pipeline.last_request
    assert result.ok is False
    assert request is not None and request.ok is False
    by_id = {command.device_id: command.status for command in request.commands}
    assert by_id["light.desk"] == "success"
    assert by_id["light.room"] == "execution_failed"
    assert _ids(client.calls) == {"light.desk", "light.room"}


def test_playing_and_paused_sessions_are_both_active():
    inventory = _inventory([], [
        _media("media_player.playing", state="playing"),
        _media("media_player.paused", state="paused"),
    ])
    calls, result, _ = _run(inventory, "звук")
    assert calls == []
    assert result.payload["status"] == "ambiguous"
    assert result.payload["reason"] == "multiple_active_devices"


def test_no_brightness_target_is_not_success():
    inventory = _inventory([], [_media("media_player.idle", state="off")])
    calls, result, request = _run(inventory, "сделай темнее")
    assert calls == []
    assert result.ok is False
    assert request is not None and request.ok is False
