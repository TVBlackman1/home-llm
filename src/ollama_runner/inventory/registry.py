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
    "video.play",
})
AUDIO = VOLUME | TRANSPORT | frozenset({"audio.play"})
SPEAKER = POWER | AUDIO

_CAPABILITIES = {
    "light": LIGHT,
    "tv": TV,
    "soundbar": AUDIO,
    "speaker": SPEAKER,
    "headphones": SPEAKER,
    "player": AUDIO,
    "radio": AUDIO | frozenset({"radio.play"}),
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
    ("радиоприемник", "radio"),
    ("приемник", "radio"),
    ("микрофон", "microphone"),
    ("градусник", "sensor"),
    ("термометр", "sensor"),
    ("гигрометр", "sensor"),
    ("радио", "radio"),
    ("лент", "light"),
    ("микро", "microphone"),
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
    "photos.show",
    "media.play",
    "media.pause",
    "media.resume",
    "media.stop",
    "media.next",
    "media.previous",
})

# Household names. Speakers in this home are «колонка», so «акустическая система»
# is the soundbar, the device already called «аудиосистема».
_TYPE_PHRASES = (
    ("компьютерный монитор", "monitor"),
    ("компьютерного монитора", "monitor"),
    ("светодиодную ленту", "light"),
    ("светодиодная лента", "light"),
    ("световую ленту", "light"),
    ("световая лента", "light"),
    ("диодную ленту", "light"),
    ("диодная лента", "light"),
    ("потолочную люстру", "light"),
    ("потолочная люстра", "light"),
    ("акустическую систему", "soundbar"),
    ("акустическая система", "soundbar"),
    ("акустической системы", "soundbar"),
    ("звуковую панель", "soundbar"),
    ("звуковая панель", "soundbar"),
    ("звуковой панели", "soundbar"),
    ("датчик температуры", "sensor"),
    ("датчик влажности", "sensor"),
)

_AREAS = {
    "кухня": ("кухня", "кухне", "кухню", "кухни"),
    "спальня": ("спальня", "спальне", "спальню", "спальни"),
    "гостиная": ("гостиная", "гостиной", "гостиную"),
    "коридор": ("коридор", "коридоре", "коридора"),
    "ванная": ("ванная", "ванной", "ванную"),
    "кабинет": ("кабинет", "кабинете", "кабинета", "офис", "офисе", "офиса"),
    "балкон": ("балкон", "балконе", "балкона"),
    "детская": ("детская", "детской", "детскую"),
}

_OWNER_ALIASES = {
    "me": ("я", "мой", "моя", "мои", "моё", "мое", "моего", "мою", "у меня"),
    "mama": ("мама", "мамы", "маме", "маму", "мамин", "мамина", "мамино", "маминого", "мамину", "у мамы"),
    "papa": ("папа", "папы", "папе", "папу", "папин", "папина", "папино", "у папы"),
    "masha": ("маша", "маши", "маше", "машу", "машин", "машина", "машину", "у маши"),
    "anton": ("антон", "антона", "антону", "антонов", "антона", "у антона"),
    "andrey": ("андрей", "андрея", "андрею", "андреев", "у андрея"),
    "artem": ("артем", "артема", "артему", "артемов", "у артема"),
    "marina": ("марина", "марины", "марине", "марину", "маринин", "у марины"),
    "common": ("общий", "общая", "общее", "общие"),
}

_ENDINGS = (
    "ого", "ему", "ому", "ами", "ями", "ах", "ях", "ам", "ям",
    "ую", "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ий", "ый", "ою", "ею",
    "ом", "ем", "ов", "ев", "ию", "ью", "ия", "ья",
    "а", "я", "у", "ю", "ы", "и", "е", "о",
)


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


def fold(text: str) -> str:
    return text.casefold().replace("ё", "е")


def stem_token(token: str) -> str:
    for ending in _ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 3:
            return token[: -len(ending)]
    return token


def token_stems(text: str) -> set[str]:
    tokens = re.findall(r"[0-9a-zа-яе]+", fold(text))
    return {stem_token(token) for token in tokens}


def _bounded(alias: str) -> re.Pattern[str]:
    return re.compile(
        r"(?<![0-9a-zа-яе])" + re.escape(fold(alias)) + r"(?![0-9a-zа-яе])"
    )


def canonicalize_device_type(value: str | None) -> str | None:
    if value is None:
        return None
    key = fold(value.strip())
    if not key:
        return None
    phrase = device_type_from_text(key)
    if phrase is not None and phrase[1] == key:
        return phrase[0]
    return _TYPE_ALIASES.get(key)


def device_type_from_text(text: str) -> tuple[str, str] | None:
    folded = fold(text)
    best: tuple[int, str, str] | None = None
    for phrase, type_name in _TYPE_PHRASES:
        match = _bounded(phrase).search(folded)
        if match is None:
            continue
        raw = text[match.start():match.end()]
        if best is None or len(phrase) > best[0]:
            best = (len(phrase), type_name, raw)
    if best is not None:
        return best[1], best[2]

    tokens = re.findall(r"[0-9a-zа-яе]+", folded)
    raw_tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text)
    stem_hit: tuple[int, str, str] | None = None
    for index, token in enumerate(tokens):
        exact = _EXACT_TOKENS.get(token)
        if exact and (stem_hit is None or len(token) > stem_hit[0]):
            raw = raw_tokens[index] if index < len(raw_tokens) else token
            stem_hit = (len(token), exact, raw)
        for stem, type_name in _TYPE_STEMS:
            if token.startswith(stem) and (stem_hit is None or len(stem) > stem_hit[0]):
                raw = raw_tokens[index] if index < len(raw_tokens) else token
                stem_hit = (len(stem), type_name, raw)
    if stem_hit is None:
        return None
    return stem_hit[1], stem_hit[2]


def area_from_text(text: str) -> tuple[str, str] | None:
    folded = fold(text)
    best: tuple[int, str, str] | None = None
    for canonical, aliases in _AREAS.items():
        for alias in aliases:
            match = _bounded(alias).search(folded)
            if match is None:
                continue
            if best is None or len(alias) > best[0]:
                best = (len(alias), canonical, text[match.start():match.end()])
    if best is None:
        return None
    return best[1], best[2]


def canonicalize_area(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    found = area_from_text(value)
    if found is not None:
        return found[0]
    folded = fold(value.strip())
    for canonical in _AREAS:
        if fold(canonical) == folded:
            return canonical
    return None


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
            self._owner_by_key[fold(owner.id)] = owner
            self._owner_by_key[fold(owner.name)] = owner
            for alias in _OWNER_ALIASES.get(owner.id, ()):
                self._owner_by_key[fold(alias)] = owner

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
        owner = self._owner_by_key.get(fold(name.strip()))
        return owner.id if owner else None

    def owner_name(self, owner_id: str | None) -> str:
        if not owner_id:
            return ""
        owner = self._owner_by_key.get(fold(owner_id))
        return owner.name if owner else owner_id

    def canonicalize_owner(self, name: str | None) -> str | None:
        owner_id = self.owner_id_for(name)
        if owner_id is None:
            return None
        return self.owner_name(owner_id) or None

    def owner_from_text(self, text: str) -> tuple[str, str] | None:
        folded = fold(text)
        best: tuple[int, str, str] | None = None
        for owner in self._owners:
            aliases = (owner.name, *_OWNER_ALIASES.get(owner.id, ()))
            for alias in aliases:
                match = _bounded(alias).search(folded)
                if match is None:
                    continue
                if best is None or len(alias) > best[0]:
                    best = (len(alias), owner.id, text[match.start():match.end()])
        if best is None:
            return None
        return self.owner_name(best[1]), best[2]

    def unique_alias_type(self, text: str) -> str | None:
        stems = token_stems(text)
        if not stems:
            return None
        types: set[str] = set()
        for device in self._devices:
            for alias in device.aliases:
                alias_stems = token_stems(alias)
                if alias_stems and alias_stems <= stems:
                    types.add(device.type)
        if len(types) == 1:
            return next(iter(types))
        return None

    def alias_mentioned(self, text: str) -> bool:
        stems = token_stems(text)
        if not stems:
            return False
        for device in self._devices:
            for alias in device.aliases:
                alias_stems = token_stems(alias)
                if alias_stems and alias_stems <= stems:
                    return True
        return False

    def with_device(self, device: RegistryDevice) -> DeviceRegistry:
        devices = [item for item in self._devices if item.id != device.id]
        devices.append(device)
        return DeviceRegistry(devices, self._owners)
