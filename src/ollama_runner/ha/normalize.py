from __future__ import annotations

from dataclasses import dataclass

from ollama_runner.inventory.registry import (
    DeviceRegistry,
    RegistryDevice,
    canonicalize_area,
    fold,
    trailing_index,
)
from ollama_runner.semantic import DeviceRuntime


_LIGHT_BRIGHTNESS = frozenset({
    "brightness.increase",
    "brightness.decrease",
    "brightness.set",
})
_COLOR_MODES = frozenset({"hs", "rgb", "xy", "rgbw", "rgbww", "color_temp"})
_TV_BITS = (
    (128, "device.turn_on"),
    (256, "device.turn_off"),
    (16384, "media.play"),
    (1, "media.pause"),
    (4096, "media.stop"),
    (4, "volume.set"),
    (1024, "volume.increase"),
    (1024, "volume.decrease"),
    (32, "media.next"),
    (16, "media.previous"),
)


@dataclass(frozen=True)
class Binding:
    """How a registry id is executed in Home Assistant."""

    entity_id: str
    domain: str
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class HomeInventory:
    registry: DeviceRegistry
    bindings: dict[str, Binding]
    state: dict[str, DeviceRuntime]


def load_inventory(client) -> HomeInventory:
    """Read the current Home Assistant snapshot and build the resolver registry."""

    entities = client.get_entities()
    states = {row["entity_id"]: row for row in client.get_states()}
    wanted = [
        entity["entity_id"]
        for entity in entities
        if _candidate(entity)
    ]
    details = {entity_id: client.get_entity(entity_id) for entity_id in wanted}
    return build_inventory(
        areas=client.get_areas(),
        devices=client.get_devices(),
        entities=entities,
        labels=client.get_labels(),
        states=list(states.values()),
        services=client.get_services(),
        entity_details=details,
    )


def build_inventory(
    *,
    areas: list[dict],
    devices: list[dict],
    entities: list[dict],
    labels: list[dict],
    states: list[dict],
    services: list[dict],
    entity_details: dict[str, dict],
) -> HomeInventory:
    areas_by_id = {area["area_id"]: area for area in areas}
    devices_by_id = {device["id"]: device for device in devices}
    labels_by_id = {_label_id(label): label for label in labels}
    states_by_id = {row["entity_id"]: row for row in states}
    light_services = _service_names(services, "light")
    records: list[RegistryDevice] = []
    bindings: dict[str, Binding] = {}
    runtime: dict[str, DeviceRuntime] = {}

    for entity in entities:
        entity_id = entity.get("entity_id") or ""
        if not _candidate(entity):
            continue
        detail = entity_details.get(entity_id) or {}
        state = states_by_id.get(entity_id) or {}
        domain = entity_id.split(".", 1)[0]
        if domain == "media_player" and not _is_tv(state, detail):
            continue
        device = devices_by_id.get(entity.get("device_id") or "")
        area_id = entity.get("area_id") or (device or {}).get("area_id")
        attrs = state.get("attributes") or {}
        members = _members(attrs)
        label_ids = list(entity.get("labels") or [])
        if device is not None:
            label_ids.extend(device.get("labels") or [])
        records.append(
            RegistryDevice(
                id=entity_id,
                type="tv" if domain == "media_player" else "light",
                owner_id=_owner_id(label_ids, labels_by_id),
                area=_area_name(areas_by_id.get(area_id)),
                aliases=_aliases(device, state, detail or entity),
                capabilities=_capabilities(domain, attrs, detail, light_services),
                ordinal=_ordinal(device, state, detail or entity),
            )
        )
        bindings[entity_id] = Binding(
            entity_id=entity_id,
            domain=domain,
            members=members,
        )
        runtime[entity_id] = _runtime(domain, state.get("state") or "")

    return HomeInventory(
        registry=DeviceRegistry(records),
        bindings=bindings,
        state=runtime,
    )


def _candidate(entity: dict) -> bool:
    if entity.get("disabled_by") or entity.get("hidden_by"):
        return False
    if entity.get("entity_category") in {"config", "diagnostic"}:
        return False
    entity_id = entity.get("entity_id") or ""
    domain = entity_id.split(".", 1)[0]
    return domain in {"light", "media_player"}


def _is_tv(state: dict, detail: dict) -> bool:
    attrs = state.get("attributes") or {}
    return (
        attrs.get("device_class") == "tv"
        or detail.get("device_class") == "tv"
        or detail.get("original_device_class") == "tv"
    )


def _area_name(area: dict | None) -> str:
    if not area:
        return ""
    aliases = area.get("aliases") or []
    for candidate in (*aliases, area.get("name")):
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        canonical = canonicalize_area(candidate)
        if canonical:
            return canonical
    name = area.get("name")
    return name.strip() if isinstance(name, str) else ""


def _owner_id(label_ids: list, labels_by_id: dict[str, dict]) -> str:
    found: list[str] = []
    for label_id in label_ids:
        record = labels_by_id.get(label_id) or {}
        name = record.get("name") or ""
        if not isinstance(name, str) or not name.startswith("owner:"):
            continue
        owner = name.split(":", 1)[1].strip()
        if owner and owner not in found:
            found.append(owner)
    if len(found) == 1:
        return found[0]
    return ""


def _ordinal(device: dict | None, state: dict, entity: dict) -> int | None:
    """One trailing 1–4 agreed by the same names that become aliases.

    The integration's product name is not scanned. It often ends in a model
    digit that is not the user's index. Entity ids are not scanned either.
    """

    attrs = state.get("attributes") or {}
    values = [
        (device or {}).get("name_by_user"),
        attrs.get("friendly_name"),
        entity.get("original_name"),
        entity.get("name"),
    ]
    for alias in entity.get("aliases") or []:
        values.append(alias)
    found: list[int] = []
    for value in values:
        if not isinstance(value, str):
            continue
        number = trailing_index(value)
        if number is not None and number not in found:
            found.append(number)
    if len(found) == 1:
        return found[0]
    return None


def _aliases(device: dict | None, state: dict, entity: dict) -> tuple[str, ...]:
    attrs = state.get("attributes") or {}
    values = [
        (device or {}).get("name_by_user"),
        attrs.get("friendly_name"),
        entity.get("original_name"),
        entity.get("name"),
    ]
    for alias in entity.get("aliases") or []:
        values.append(alias)
    seen: set[str] = set()
    names: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        key = fold(text)
        if key in seen:
            continue
        seen.add(key)
        names.append(text)
    return tuple(names)


def _capabilities(
    domain: str,
    attrs: dict,
    detail: dict,
    light_services: set[str],
) -> frozenset[str]:
    if domain == "light":
        caps = set()
        if "turn_on" in light_services:
            caps.add("device.turn_on")
        if "turn_off" in light_services:
            caps.add("device.turn_off")
        modes = set(attrs.get("supported_color_modes") or [])
        extra = (detail.get("capabilities") or {}) if detail else {}
        modes.update(extra.get("supported_color_modes") or [])
        if modes - {"onoff"}:
            caps.update(_LIGHT_BRIGHTNESS)
        if modes & _COLOR_MODES:
            caps.add("color.set")
        return frozenset(caps)
    features = attrs.get("supported_features")
    if not isinstance(features, int):
        return frozenset()
    return frozenset(intent for bit, intent in _TV_BITS if features & bit)


def _members(attrs: dict) -> tuple[str, ...]:
    raw = attrs.get("entity_id")
    if not isinstance(raw, list):
        return ()
    return tuple(entity_id for entity_id in raw if isinstance(entity_id, str))


def _runtime(domain: str, state: str) -> DeviceRuntime:
    if state in {"unavailable", "unknown", ""}:
        return DeviceRuntime(power="unavailable")
    if domain == "media_player":
        media = state if state in {"playing", "paused"} else None
        power = "off" if state == "off" else "on"
        return DeviceRuntime(media=media, power=power)
    return DeviceRuntime(power=state if state in {"on", "off"} else None)


def _service_names(services: list[dict], domain: str) -> set[str]:
    for row in services:
        if row.get("domain") == domain:
            return set((row.get("services") or {}))
    return set()


def _label_id(label: dict) -> str:
    return str(label.get("label_id") or label.get("id") or "")
