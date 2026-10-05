from ollama_runner.ha.execute import HaExecutor, planned_call
from ollama_runner.ha.normalize import Binding, build_inventory
from ollama_runner.semantic import ResolvedCommand
from ollama_runner.sinks.jsonprint import PrintSink


def _area(area_id: str, name: str, aliases: list[str] | None = None) -> dict:
    return {"area_id": area_id, "name": name, "aliases": aliases or []}


def _light(entity_id: str, *, device_id: str | None, area_id: str | None, labels: list[str] | None = None) -> dict:
    return {
        "entity_id": entity_id,
        "device_id": device_id,
        "area_id": area_id,
        "platform": "tuya_local",
        "disabled_by": None,
        "hidden_by": None,
        "entity_category": None,
        "labels": labels or [],
        "name": None,
        "original_name": None,
        "aliases": [],
    }


def _snapshot():
    areas = [
        _area("living_room", "Living Room", ["гостиная"]),
        _area("kitchen", "Kitchen"),
        _area("bedroom", "Bedroom", ["спальня"]),
    ]
    devices = [
        {"id": "lamp-1", "name": "Bulb", "name_by_user": "Лампа 1", "area_id": "living_room", "labels": ["label-me"]},
        {"id": "lamp-2", "name": "Bulb 2", "name_by_user": "Лампа 2", "area_id": "living_room", "labels": []},
        {"id": "tv", "name": "Samsung The Frame", "name_by_user": None, "area_id": "living_room", "labels": []},
    ]
    entities = [
        _light("light.bulb_e27_lemon_3", device_id="lamp-1", area_id=None),
        _light("light.bulb_e27_lemon_3_2", device_id="lamp-2", area_id=None),
        {
            "entity_id": "light.lampa",
            "device_id": None,
            "area_id": "living_room",
            "platform": "group",
            "disabled_by": None,
            "hidden_by": None,
            "entity_category": None,
            "labels": [],
            "name": None,
            "original_name": "Лампа",
            "aliases": [None],
        },
        {
            "entity_id": "media_player.frame",
            "device_id": "tv",
            "area_id": None,
            "platform": "samsungtv",
            "disabled_by": None,
            "hidden_by": None,
            "entity_category": None,
            "labels": [],
            "name": None,
            "original_name": None,
            "aliases": ["телевизор"],
        },
        {
            "entity_id": "remote.frame",
            "device_id": "tv",
            "area_id": "living_room",
            "platform": "samsungtv",
            "disabled_by": None,
            "hidden_by": None,
            "entity_category": None,
            "labels": [],
            "name": None,
            "original_name": None,
            "aliases": [],
        },
        {
            "entity_id": "switch.permit",
            "device_id": None,
            "area_id": None,
            "platform": "mqtt",
            "disabled_by": None,
            "hidden_by": None,
            "entity_category": None,
            "labels": [],
            "name": None,
            "original_name": "Permit join",
            "aliases": [],
        },
    ]
    states = [
        {
            "entity_id": "light.bulb_e27_lemon_3",
            "state": "on",
            "attributes": {
                "friendly_name": "Лампа 1",
                "supported_color_modes": ["color_temp", "hs"],
                "min_color_temp_kelvin": 2700,
                "max_color_temp_kelvin": 6500,
            },
        },
        {
            "entity_id": "light.bulb_e27_lemon_3_2",
            "state": "off",
            "attributes": {"friendly_name": "Лампа 2", "supported_color_modes": ["hs"]},
        },
        {
            "entity_id": "light.lampa",
            "state": "on",
            "attributes": {
                "friendly_name": "Лампа",
                "supported_color_modes": ["color_temp", "hs"],
                "min_color_temp_kelvin": 2700,
                "max_color_temp_kelvin": 6500,
                "entity_id": ["light.bulb_e27_lemon_3", "light.bulb_e27_lemon_3_2"],
            },
        },
        {
            "entity_id": "media_player.frame",
            "state": "off",
            "attributes": {
                "friendly_name": "Samsung The Frame",
                "device_class": "tv",
                "supported_features": 24509,
            },
        },
        {
            "entity_id": "remote.frame",
            "state": "off",
            "attributes": {"friendly_name": "Samsung The Frame", "supported_features": 0},
        },
    ]
    services = [
        {"domain": "light", "services": {"turn_on": {}, "turn_off": {}, "toggle": {}}},
        {"domain": "media_player", "services": {"turn_on": {}, "turn_off": {}}},
    ]
    details = {
        "light.bulb_e27_lemon_3": {"aliases": [], "capabilities": {"supported_color_modes": ["color_temp", "hs"]}},
        "light.bulb_e27_lemon_3_2": {"aliases": [], "capabilities": {}},
        "light.lampa": {"aliases": [None], "original_name": "Лампа", "capabilities": {}},
        "media_player.frame": {"aliases": ["телевизор"], "original_device_class": "tv", "capabilities": {}},
    }
    labels = [{"label_id": "label-me", "name": "owner:me"}]
    return build_inventory(
        areas=areas,
        devices=devices,
        entities=entities,
        labels=labels,
        states=states,
        services=services,
        entity_details=details,
    )


def test_lights_tv_group_and_remote():
    inventory = _snapshot()
    ids = {device.id for device in inventory.registry.devices()}
    assert ids == {
        "light.bulb_e27_lemon_3",
        "light.bulb_e27_lemon_3_2",
        "light.lampa",
        "media_player.frame",
    }

    lamp = inventory.registry.get("light.bulb_e27_lemon_3")
    assert lamp is not None
    assert lamp.type == "light"
    assert lamp.area == "гостиная"
    assert lamp.owner_id == "me"
    assert lamp.ordinal == 1
    assert "Лампа 1" in lamp.aliases
    assert "device.turn_on" in lamp.capabilities
    assert "brightness.increase" in lamp.capabilities
    assert "color.set" in lamp.capabilities
    assert inventory.bindings["light.bulb_e27_lemon_3"] == Binding(
        "light.bulb_e27_lemon_3",
        "light",
        min_color_temp_kelvin=2700,
        max_color_temp_kelvin=6500,
        color_modes=frozenset({"color_temp", "hs"}),
    )
    assert "color_temperature.set" in lamp.capabilities
    assert inventory.state["light.bulb_e27_lemon_3"].power == "on"

    other = inventory.registry.get("light.bulb_e27_lemon_3_2")
    assert other is not None
    assert other.owner_id == ""
    assert other.ordinal == 2
    assert other.area == "гостиная"
    assert "color_temperature.set" not in other.capabilities

    group = inventory.registry.get("light.lampa")
    assert group is not None
    assert group.type == "light"
    assert group.ordinal is None
    binding = inventory.bindings["light.lampa"]
    assert binding.members == ("light.bulb_e27_lemon_3", "light.bulb_e27_lemon_3_2")
    assert binding.domain == "light"

    tv = inventory.registry.get("media_player.frame")
    assert tv is not None
    assert tv.type == "tv"
    assert tv.ordinal is None
    assert "device.turn_on" in tv.capabilities
    assert "media.pause" in tv.capabilities
    assert "media.seek_forward" not in tv.capabilities
    assert "телевизор" in tv.aliases
    assert inventory.state["media_player.frame"].power == "off"
    assert inventory.registry.get("remote.frame") is None
    assert inventory.bindings["media_player.frame"].power_entity_id == "remote.frame"
    assert inventory.bindings["media_player.frame"].entity_id == "media_player.frame"


def test_power_mapping_uses_the_binding_domain():
    call = planned_call("device.turn_on", Binding("light.lampa", "light"))
    assert call is not None
    assert (call.domain, call.service, call.entity_id) == ("light", "turn_on", "light.lampa")
    off = planned_call("device.turn_off", Binding("media_player.frame", "media_player"))
    assert off is not None
    assert off.service == "turn_off"
    brighter = planned_call("brightness.increase", Binding("light.lampa", "light"))
    assert brighter is not None
    assert brighter.brightness_step_pct == 10
    assert planned_call("brightness.set", Binding("media_player.frame", "media_player"), "50") is None
    assert planned_call("device.turn_on", None) is None


class _FakeHa:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def call_service(self, domain: str, service: str, data: dict) -> None:
        self.calls.append((domain, service, data))


def test_unresolved_command_does_not_call_service():
    inventory = _snapshot()
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    resolved = ResolvedCommand(
        status="not_found",
        intent="device.turn_on",
        semantic_target_id=None,
        execution_target_id=None,
        arguments={},
        candidates=(),
    )
    result = executor.execute(resolved, PrintSink())
    assert result.ok is False
    assert client.calls == []
    assert executor.planned == []


def test_resolved_turn_on_calls_the_bound_entity():
    inventory = _snapshot()
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    resolved = ResolvedCommand(
        status="resolved",
        intent="device.turn_on",
        semantic_target_id="light.bulb_e27_lemon_3",
        execution_target_id="light.bulb_e27_lemon_3",
        arguments={},
        candidates=("light.bulb_e27_lemon_3",),
    )
    result = executor.execute(resolved, PrintSink())
    assert result.ok is True
    assert client.calls == [("light", "turn_on", {"entity_id": "light.bulb_e27_lemon_3"})]


def test_resolved_turn_off_calls_light_turn_off():
    inventory = _snapshot()
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    resolved = ResolvedCommand(
        status="resolved",
        intent="device.turn_off",
        semantic_target_id="light.lampa",
        execution_target_id="light.lampa",
        arguments={},
        candidates=("light.lampa",),
    )
    result = executor.execute(resolved, PrintSink())
    assert result.ok is True
    assert client.calls == [("light", "turn_off", {"entity_id": "light.lampa"})]


def test_unknown_color_set_does_not_call_service():
    inventory = _snapshot()
    client = _FakeHa()
    executor = HaExecutor(inventory.registry, inventory.bindings, client)
    resolved = ResolvedCommand(
        status="resolved",
        intent="color.set",
        semantic_target_id="light.bulb_e27_lemon_3",
        execution_target_id="light.bulb_e27_lemon_3",
        arguments={"value": "10"},
        candidates=("light.bulb_e27_lemon_3",),
    )
    result = executor.execute(resolved, PrintSink())
    assert result.ok is False
    assert result.payload["status"] == "unsupported"
    assert result.payload["reason"] == "color_unsupported"
    assert client.calls == []


def test_english_area_name_stays_when_no_known_alias():
    built = build_inventory(
        areas=[_area("kitchen", "Kitchen")],
        devices=[],
        entities=[{
            "entity_id": "light.kitchen",
            "device_id": None,
            "area_id": "kitchen",
            "platform": "group",
            "disabled_by": None,
            "hidden_by": None,
            "entity_category": None,
            "labels": [],
            "name": None,
            "original_name": "Kitchen",
            "aliases": [],
        }],
        labels=[],
        states=[{
            "entity_id": "light.kitchen",
            "state": "off",
            "attributes": {"friendly_name": "Kitchen", "supported_color_modes": ["onoff"]},
        }],
        services=[{"domain": "light", "services": {"turn_on": {}, "turn_off": {}}}],
        entity_details={"light.kitchen": {"aliases": [], "capabilities": {"supported_color_modes": ["onoff"]}}},
    )
    device = built.registry.get("light.kitchen")
    assert device is not None
    assert device.area == "Kitchen"
    assert "brightness.increase" not in device.capabilities
    assert "device.turn_on" in device.capabilities
