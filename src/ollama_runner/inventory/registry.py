from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ollama_runner.inventory.static import DEVICES, OWNERS
from ollama_runner.types import Owner


POWER = frozenset({"device.turn_on", "device.turn_off"})
LIGHT = POWER | frozenset({
    "brightness.increase",
    "brightness.decrease",
    "brightness.set",
    "color.set",
})
TRANSPORT = frozenset({
    "media.play",
    "media.pause",
    "media.resume",
    "media.stop",
})
VOLUME = frozenset({
    "volume.increase",
    "volume.decrease",
    "volume.set",
})
TV = POWER | TRANSPORT | frozenset({
    "media.next",
    "media.previous",
    "media.seek_forward",
    "media.seek_backward",
    "content.play",
})
AUDIO = VOLUME | TRANSPORT
SPEAKER = POWER | AUDIO

_CAPABILITIES = {
    "light": LIGHT,
    "tv": TV,
    "soundbar": AUDIO,
    "speaker": SPEAKER,
    "headphones": SPEAKER,
    "player": AUDIO,
    "radio": AUDIO,
    "projector": POWER | TRANSPORT,
    "monitor": POWER,
    "microphone": POWER,
    "vacuum": POWER,
    "climate": POWER,
    "sensor": frozenset(),
}

_PREFIXES = (
    ("floor_lamp_", "light"),
    ("led_strip_", "light"),
    ("chandelier_", "light"),
    ("humidifier_", "climate"),
    ("hygrometer_", "sensor"),
    ("thermometer_", "sensor"),
    ("thermostat_", "climate"),
    ("conditioner_", "climate"),
    ("microphone", "microphone"),
    ("headphones", "headphones"),
    ("projector", "projector"),
    ("soundbar", "soundbar"),
    ("purifier_", "climate"),
    ("speaker_", "speaker"),
    ("monitor", "monitor"),
    ("vacuum_", "vacuum"),
    ("player", "player"),
    ("radio", "radio"),
    ("light_", "light"),
    ("lamp_", "light"),
    ("fan_", "climate"),
    ("tv", "tv"),
)

_RELATIONSHIPS: dict[str, tuple[tuple[str, str], ...]] = {
    "tv": (("audio_output", "soundbar"),),
    "tv_living": (("audio_output", "soundbar_living"),),
    "tv_kitchen": (("audio_output", "soundbar_kitchen"),),
    "tv_bedroom": (("audio_output", "soundbar"),),
    "tv_masha_bedroom": (("audio_output", "soundbar"),),
}

_TYPE_ALIASES = {
    "light": "light",
    "свет": "light",
    "лампа": "light",
    "освещение": "light",
    "люстра": "light",
    "торшер": "light",
    "светильник": "light",
    "tv": "tv",
    "телевизор": "tv",
    "телик": "tv",
    "телек": "tv",
    "тв": "tv",
    "ящик": "tv",
    "soundbar": "soundbar",
    "саундбар": "soundbar",
    "аудиосистема": "soundbar",
    "speaker": "speaker",
    "колонка": "speaker",
    "headphones": "headphones",
    "наушники": "headphones",
    "уши": "headphones",
    "гарнитура": "headphones",
    "monitor": "monitor",
    "монитор": "monitor",
    "player": "player",
    "плеер": "player",
    "проигрыватель": "player",
    "radio": "radio",
    "радио": "radio",
    "projector": "projector",
    "проектор": "projector",
}

_TYPE_STEMS = (
    ("колонк", "speaker"),
    ("телевиз", "tv"),
    ("саундбар", "soundbar"),
    ("аудиосистем", "soundbar"),
    ("наушник", "headphones"),
    ("гарнитур", "headphones"),
    ("монитор", "monitor"),
    ("проектор", "projector"),
    ("проигрывател", "player"),
    ("светильник", "light"),
    ("освещен", "light"),
    ("телевиз", "tv"),
    ("люстр", "light"),
    ("торшер", "light"),
    ("ламп", "light"),
    ("свет", "light"),
    ("телик", "tv"),
    ("телек", "tv"),
    ("плеер", "player"),
    ("радио", "radio"),
)

_EXACT_TOKENS = {
    "тв": "tv",
    "уши": "headphones",
    "ящик": "tv",
}

_ORDINAL_STEMS = (
    ("перв", 1),
    ("втор", 2),
    ("трет", 3),
    ("четвер", 4),
)

_VOLUME_INTENTS = frozenset({
    "volume.increase",
    "volume.decrease",
    "volume.set",
})
_RENDER_INTENTS = frozenset({
    "content.play",
    "photos.show",
    "media.play",
    "media.pause",
    "media.resume",
    "media.stop",
    "media.next",
    "media.previous",
})


@dataclass(frozen=True)
class RegistryDevice:
    id: str
    type: str
    owner_id: str
    area: str
    aliases: tuple[str, ...]
    capabilities: frozenset[str]
    ordinal: int | None = None
    relationships: tuple[tuple[str, str], ...] = ()


def infer_type(device_id: str) -> str:
    for prefix, type_name in sorted(_PREFIXES, key=lambda item: len(item[0]), reverse=True):
        if device_id.startswith(prefix):
            return type_name
    raise ValueError(f"unknown device id: {device_id}")


def infer_ordinal(device_id: str) -> int | None:
    if "speaker_first" in device_id:
        return 1
    if "speaker_second" in device_id:
        return 2
    if "speaker_third" in device_id:
        return 3
    return None


def capabilities_for(device_type: str) -> frozenset[str]:
    try:
        return _CAPABILITIES[device_type]
    except KeyError as exc:
        raise ValueError(f"unknown device type: {device_type}") from exc


def canonicalize_device_type(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().casefold().replace("ё", "е")
    if not key:
        return None
    return _TYPE_ALIASES.get(key)


def type_from_mention(mention: str | None) -> str | None:
    if not mention:
        return None
    tokens = re.findall(r"[0-9a-zа-яе]+", mention.casefold().replace("ё", "е"))
    for token in tokens:
        exact = _EXACT_TOKENS.get(token)
        if exact:
            return exact
        for stem, type_name in _TYPE_STEMS:
            if token.startswith(stem):
                return type_name
    return None


def ordinal_from_mention(mention: str | None) -> int | None:
    if not mention:
        return None
    folded = mention.casefold().replace("ё", "е")
    for stem, number in _ORDINAL_STEMS:
        if stem in folded:
            return number
    return None


def relation_supports(relation_type: str, intent: str) -> bool:
    if intent in _VOLUME_INTENTS:
        return relation_type == "audio_output"
    if intent in _RENDER_INTENTS:
        return relation_type in {"media_renderer", "audio_output"}
    return False


class DeviceRegistry:
    def __init__(
        self,
        devices: Sequence[RegistryDevice],
        owners: Sequence[Owner] | None = None,
    ) -> None:
        self._devices = tuple(devices)
        self._by_id = {device.id: device for device in self._devices}
        self._owners = tuple(OWNERS if owners is None else owners)
        self._owner_by_key = {}
        for owner in self._owners:
            self._owner_by_key[owner.id.casefold()] = owner
            self._owner_by_key[owner.name.casefold()] = owner

    @classmethod
    def from_static(cls) -> DeviceRegistry:
        grouped: dict[str, list] = {}
        for device in DEVICES:
            grouped.setdefault(device.id, []).append(device)

        records = []
        for device_id, rows in grouped.items():
            device_type = infer_type(device_id)
            aliases = tuple(dict.fromkeys(row.name for row in rows))
            records.append(
                RegistryDevice(
                    id=device_id,
                    type=device_type,
                    owner_id=rows[0].owner_id,
                    area=rows[0].place,
                    aliases=aliases,
                    capabilities=capabilities_for(device_type),
                    ordinal=infer_ordinal(device_id),
                    relationships=_RELATIONSHIPS.get(device_id, ()),
                )
            )
        return cls(records, OWNERS)

    def devices(self) -> tuple[RegistryDevice, ...]:
        return self._devices

    def get(self, device_id: str) -> RegistryDevice | None:
        return self._by_id.get(device_id)

    def owner_id_for(self, name: str | None) -> str | None:
        if not name or not name.strip():
            return None
        owner = self._owner_by_key.get(name.strip().casefold())
        return owner.id if owner else None

    def owner_name(self, owner_id: str | None) -> str:
        if not owner_id:
            return ""
        owner = self._owner_by_key.get(owner_id.casefold())
        return owner.name if owner else owner_id

    def with_device(self, device: RegistryDevice) -> DeviceRegistry:
        devices = [item for item in self._devices if item.id != device.id]
        devices.append(device)
        return DeviceRegistry(devices, self._owners)
