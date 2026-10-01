from __future__ import annotations

from collections.abc import Sequence

from ollama_runner.types import Device, Owner


OWNERS = [
    Owner("common", "общий"),
    Owner("me", "я"),
    Owner("masha", "Маша"),
    Owner("marina", "Марина"),
    Owner("mama", "мама"),
    Owner("papa", "папа"),
    Owner("anton", "Антон"),
    Owner("andrey", "Андрей"),
    Owner("artem", "Артем"),
]


def _aliases(device_id: str, owner_id: str, place: str, *names: str) -> list[Device]:
    return [
        Device(device_id, name, owner_id, place)
        for name in names
    ]


DEVICES = [
    *_aliases("light_common", "common", "", "свет", "лампа", "освещение"),
    *_aliases("light_me", "me", "", "свет", "лампа", "освещение"),
    *_aliases("light_masha", "masha", "", "свет", "лампа", "освещение"),
    *_aliases("light_mama", "mama", "", "свет", "лампа", "освещение"),
    *_aliases("light_papa", "papa", "", "свет", "лампа", "освещение"),
    *_aliases("light_anton", "anton", "", "свет", "лампа", "освещение"),
    *_aliases("light_common_kitchen", "common", "кухня", "свет", "лампа", "освещение"),
    *_aliases("light_common_bedroom", "common", "спальня", "свет", "лампа", "освещение"),
    *_aliases("light_common_living", "common", "гостиная", "свет", "лампа", "освещение"),
    *_aliases("light_common_hallway", "common", "коридор", "свет", "лампа", "освещение"),
    *_aliases("light_common_bath", "common", "ванная", "свет", "лампа", "освещение"),
    *_aliases("light_me_bedroom", "me", "спальня", "свет", "лампа", "освещение"),
    *_aliases("light_me_cabinet", "me", "кабинет", "свет", "лампа", "освещение"),
    *_aliases("light_me_kitchen", "me", "кухня", "свет", "лампа", "освещение"),
    *_aliases("light_masha_bedroom", "masha", "спальня", "свет", "лампа", "освещение"),
    *_aliases("light_masha_kitchen", "masha", "кухня", "свет", "лампа", "освещение"),
    *_aliases("light_mama_bedroom", "mama", "спальня", "свет", "лампа", "освещение"),
    *_aliases("light_mama_kitchen", "mama", "кухня", "свет", "лампа", "освещение"),
    *_aliases("light_papa_cabinet", "papa", "кабинет", "свет", "лампа", "освещение"),
    *_aliases("light_anton_bedroom", "anton", "спальня", "свет", "лампа", "освещение"),
    *_aliases("light_andrey_balcony", "andrey", "балкон", "свет", "лампа", "освещение"),
    *_aliases("light_artem_nursery", "artem", "детская", "свет", "лампа", "освещение"),
    *_aliases("lamp_marina", "marina", "", "лампа", "светильник"),
    *_aliases("lamp_marina_bedroom", "marina", "спальня", "лампа", "светильник"),
    *_aliases("chandelier_common", "common", "", "люстра", "потолочная люстра"),
    *_aliases("chandelier_living", "common", "гостиная", "люстра", "потолочная люстра"),
    *_aliases("floor_lamp_anton", "anton", "", "торшер", "напольный светильник"),
    *_aliases("floor_lamp_anton_living", "anton", "гостиная", "торшер", "напольный светильник"),
    *_aliases("led_strip_andrey", "andrey", "", "светодиодная лента", "световая лента", "лента", "диодная лента"),
    *_aliases("led_strip_andrey_cabinet", "andrey", "кабинет", "светодиодная лента", "световая лента", "лента", "диодная лента"),
    *_aliases("tv", "common", "", "телевизор", "телик", "телек", "тв", "ящик"),
    *_aliases("tv_living", "common", "гостиная", "телевизор", "телик", "телек", "тв", "ящик"),
    *_aliases("tv_kitchen", "common", "кухня", "телевизор", "телик", "телек", "тв", "ящик"),
    *_aliases("tv_bedroom", "common", "спальня", "телевизор", "телик", "телек", "тв", "ящик"),
    *_aliases("tv_masha_bedroom", "masha", "спальня", "телевизор", "телик", "телек", "тв", "ящик"),
    *_aliases("soundbar", "common", "", "саундбар", "аудиосистема", "музыка"),
    *_aliases("soundbar_living", "common", "гостиная", "саундбар", "аудиосистема", "музыка"),
    *_aliases("soundbar_kitchen", "common", "кухня", "саундбар", "аудиосистема", "музыка"),
    *_aliases("player", "common", "", "плеер", "проигрыватель"),
    *_aliases("player_kitchen", "common", "кухня", "плеер", "проигрыватель"),
    *_aliases("player_bedroom", "common", "спальня", "плеер", "проигрыватель"),
    *_aliases("radio", "common", "", "радио", "радиоприемник", "приемник"),
    *_aliases("radio_kitchen", "common", "кухня", "радио", "радиоприемник", "приемник"),
    *_aliases("speaker_first", "common", "", "колонка первая"),
    *_aliases("speaker_first_living", "common", "гостиная", "колонка первая"),
    *_aliases("speaker_second", "common", "", "колонка вторая"),
    *_aliases("speaker_second_kitchen", "common", "кухня", "колонка вторая"),
    *_aliases("projector", "common", "", "проектор"),
    *_aliases("projector_living", "common", "гостиная", "проектор"),
    *_aliases("monitor", "common", "", "монитор", "компьютерный монитор"),
    *_aliases("monitor_me_cabinet", "me", "кабинет", "монитор", "компьютерный монитор"),
    *_aliases("headphones", "common", "", "наушники", "головные наушники", "гарнитура", "уши"),
    *_aliases("microphone", "common", "", "микрофон", "микро"),
    *_aliases("microphone_cabinet", "common", "кабинет", "микрофон", "микро"),
    *_aliases("thermometer_mama", "mama", "", "термометр", "датчик температуры", "градусник"),
    *_aliases("thermometer_mama_kitchen", "mama", "кухня", "термометр", "датчик температуры", "градусник"),
    *_aliases("hygrometer_papa", "papa", "", "гигрометр", "датчик влажности"),
    *_aliases("hygrometer_papa_bedroom", "papa", "спальня", "гигрометр", "датчик влажности"),
    *_aliases("thermostat_common", "common", "", "термостат"),
    *_aliases("thermostat_living", "common", "гостиная", "термостат"),
    *_aliases("fan_common", "common", "", "вентилятор"),
    *_aliases("fan_living", "common", "гостиная", "вентилятор"),
    *_aliases("conditioner_common", "common", "", "кондиционер", "кондей"),
    *_aliases("conditioner_bedroom", "common", "спальня", "кондиционер", "кондей"),
    *_aliases("humidifier_mama", "mama", "", "увлажнитель", "увлажнитель воздуха"),
    *_aliases("humidifier_mama_bedroom", "mama", "спальня", "увлажнитель", "увлажнитель воздуха"),
    *_aliases("purifier_papa", "papa", "", "очиститель воздуха"),
    *_aliases("purifier_papa_living", "papa", "гостиная", "очиститель воздуха"),
    *_aliases("vacuum_common", "common", "", "пылесос", "робот пылесос", "робот"),
]


class StaticInventory:
    version = 4

    def owners(self) -> Sequence[Owner]:
        return OWNERS

    def devices(self) -> Sequence[Device]:
        return DEVICES
