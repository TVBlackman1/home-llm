import httpx

from ollama_runner.main import run_home_request
from ollama_runner.semantic import NotCommandOutcome


class _Idle:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, text, context) -> NotCommandOutcome:
        self.calls += 1
        return NotCommandOutcome()


class _House:
    """Two sequential inventories. Discovery methods read the current one."""

    def __init__(self, first: dict, second: dict) -> None:
        self._pending = [first, second]
        self.current: dict | None = None
        self.discovery: list[str] = []
        self.services: list[tuple] = []
        self.fail = False

    def get_entities(self) -> list:
        self.discovery.append("entities")
        if self.fail:
            raise httpx.ConnectError("down")
        self.current = self._pending.pop(0)
        return self.current["entities"]

    def get_states(self) -> list:
        self.discovery.append("states")
        return self.current["states"]

    def get_entity(self, entity_id: str) -> dict:
        self.discovery.append(f"entity:{entity_id}")
        return self.current["details"].get(entity_id, {})

    def get_areas(self) -> list:
        self.discovery.append("areas")
        return self.current["areas"]

    def get_devices(self) -> list:
        self.discovery.append("devices")
        return self.current["devices"]

    def get_labels(self) -> list:
        self.discovery.append("labels")
        return []

    def get_services(self) -> list:
        self.discovery.append("services")
        return self.current["services"]

    def call_service(self, domain: str, service: str, data: dict) -> int:
        self.services.append((domain, service, data))
        return 200


def _entity(entity_id: str, device_id: str) -> dict:
    return {
        "entity_id": entity_id,
        "device_id": device_id,
        "area_id": None,
        "disabled_by": None,
        "hidden_by": None,
        "entity_category": None,
        "labels": [],
        "aliases": [],
    }


def _light(name: str) -> dict:
    """Two lights, so a turn-on resolves only when the spoken alias matches."""

    return {
        "areas": [{"area_id": "room", "name": "Room", "aliases": []}],
        "devices": [
            {"id": "lamp", "name": name, "name_by_user": name, "area_id": "room", "labels": []},
            {"id": "other", "name": "Люстра", "name_by_user": "Люстра", "area_id": "room", "labels": []},
        ],
        "entities": [_entity("light.lamp", "lamp"), _entity("light.other", "other")],
        "states": [
            {
                "entity_id": "light.lamp",
                "state": "off",
                "attributes": {"friendly_name": name, "supported_color_modes": ["onoff"]},
            },
            {
                "entity_id": "light.other",
                "state": "off",
                "attributes": {"friendly_name": "Люстра", "supported_color_modes": ["onoff"]},
            },
        ],
        "details": {
            "light.lamp": {"aliases": [name]},
            "light.other": {"aliases": ["Люстра"]},
        },
        "services": [{"domain": "light", "services": {"turn_on": {}, "turn_off": {}}}],
    }


def _media(entity_id: str, device_id: str, *, features: int, state: str, remote: str | None) -> dict:
    entities = [_entity(entity_id, device_id)]
    if remote:
        entities.append(_entity(remote, device_id))
    return {
        "entity": entities,
        "state": {
            "entity_id": entity_id,
            "state": state,
            "attributes": {
                "friendly_name": entity_id,
                "device_class": "tv",
                "supported_features": features,
            },
        },
        "device": {"id": device_id, "name": entity_id, "name_by_user": None, "area_id": "room", "labels": []},
        "detail": {"aliases": [], "original_device_class": "tv"},
    }


def _tvs(*players: dict) -> dict:
    entities = []
    states = []
    devices = []
    details = {}
    for player in players:
        entities.extend(player["entity"])
        states.append(player["state"])
        devices.append(player["device"])
        details[player["state"]["entity_id"]] = player["detail"]
    return {
        "areas": [{"area_id": "room", "name": "Room", "aliases": []}],
        "devices": devices,
        "entities": entities,
        "states": states,
        "details": details,
        "services": [{"domain": "media_player", "services": {"turn_on": {}, "turn_off": {}}}],
    }


def _ask(house: _House, text: str, nlu: _Idle):
    return run_home_request(house, text, nlu=nlu)


def test_second_request_uses_the_new_alias():
    house = _House(_light("ночник"), _light("бра"))
    nlu = _Idle()
    first = _ask(house, "Включи ночник", nlu)
    second = _ask(house, "Включи бра", nlu)
    assert first.ok is True
    assert second.ok is True
    assert house.services == [
        ("light", "turn_on", {"entity_id": "light.lamp"}),
        ("light", "turn_on", {"entity_id": "light.lamp"}),
    ]
    assert nlu.calls == 0


def test_second_request_sees_a_newly_advertised_capability():
    quiet = 128 | 256 | 1
    loud = quiet | 1024
    tv = lambda features: _tvs(_media(
        "media_player.frame",
        "tv",
        features=features,
        state="on",
        remote="remote.frame",
    ))
    house = _House(tv(quiet), tv(loud))
    nlu = _Idle()
    missing = _ask(house, "Сделай телевизор громче", nlu)
    assert missing.ok is False
    assert missing.payload["status"] == "unsupported"
    assert house.services == []
    present = _ask(house, "Сделай телевизор громче", nlu)
    assert present.ok is True
    assert house.services == [("media_player", "volume_up", {"entity_id": "media_player.frame"})]
    assert nlu.calls == 0


def test_second_request_resolves_the_newly_playing_device():
    def house_for(playing: str, idle: str) -> dict:
        return _tvs(
            _media(playing, "playing", features=1, state="playing", remote=None),
            _media(idle, "idle", features=1, state="off", remote=None),
        )
    first_map = house_for("media_player.one", "media_player.two")
    second_map = house_for("media_player.two", "media_player.one")
    house = _House(first_map, second_map)
    nlu = _Idle()
    _ask(house, "Поставь на паузу", nlu)
    _ask(house, "Поставь на паузу", nlu)
    assert house.services == [
        ("media_player", "media_pause", {"entity_id": "media_player.one"}),
        ("media_player", "media_pause", {"entity_id": "media_player.two"}),
    ]
    assert nlu.calls == 0


def test_second_request_toggles_the_new_remote():
    def tv(remote: str) -> dict:
        return _tvs(_media(
            "media_player.frame",
            "tv",
            features=128,
            state="off",
            remote=remote,
        ))
    house = _House(tv("remote.alpha"), tv("remote.beta"))
    nlu = _Idle()
    _ask(house, "Включи телевизор", nlu)
    _ask(house, "Включи телевизор", nlu)
    assert house.services == [
        ("homeassistant", "toggle", {"entity_id": "remote.alpha"}),
        ("homeassistant", "toggle", {"entity_id": "remote.beta"}),
    ]
    assert house.discovery.count("states") > 2
    assert nlu.calls == 0


def test_a_failed_refresh_does_not_reuse_the_previous_map():
    house = _House(_light("ночник"), _light("бра"))
    nlu = _Idle()
    assert _ask(house, "Включи ночник", nlu).ok is True
    house.fail = True
    failed = _ask(house, "Включи ночник", nlu)
    assert failed.ok is False
    assert failed.payload["status"] == "unavailable"
    assert failed.payload["reason"] == "ha_connection_failed"
    assert house.services == [("light", "turn_on", {"entity_id": "light.lamp"})]


def test_inventory_failure_does_not_copy_the_exception_text():
    class _Secret:
        def get_entities(self):
            raise RuntimeError("Authorization: Bearer super-secret-token")

    failed = run_home_request(_Secret(), "Включи свет", nlu=_Idle())
    assert failed.payload["status"] == "unavailable"
    assert failed.payload["reason"] == "execution_exception:RuntimeError"
    assert "super-secret-token" not in failed.payload["reason"]
