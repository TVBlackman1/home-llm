from __future__ import annotations

import json
from pathlib import Path

from ollama_runner.inventory.static import DEVICES, OWNERS
from ollama_runner.types import Device


ROOT = Path(__file__).parent
OWNER_NAME = {owner.id: owner.name for owner in OWNERS}

PLACE_PREP = {
    "кухня": "на кухне",
    "спальня": "в спальне",
    "гостиная": "в гостиной",
    "зал": "в зале",
    "ванная": "в ванной",
    "ванная комната": "в ванной комнате",
    "коридор": "в коридоре",
    "кабинет": "в кабинете",
    "офис": "в офисе",
    "детская": "в детской",
    "балкон": "на балконе",
}

OWNER_PHRASE = {
    "": "",
    "общий": "",
    "я": "мой",
    "мама": "мамин",
    "папа": "папин",
    "Маша": "у Маши",
    "Мария": "у Марии",
    "Марина": "у Марины",
    "Антон": "у Антона",
    "Андрей": "у Андрея",
    "Артем": "у Артема",
}

DEVICE_PHRASE = {
    "колонка первая": "первую колонку",
    "колонка вторая": "вторую колонку",
    "светодиодная лента": "светодиодную ленту",
    "световая лента": "световую ленту",
    "диодная лента": "диодную ленту",
    "напольный светильник": "напольный светильник",
    "потолочная люстра": "потолочную люстру",
    "очиститель воздуха": "очиститель воздуха",
    "увлажнитель воздуха": "увлажнитель воздуха",
    "датчик температуры": "датчик температуры",
    "датчик влажности": "датчик влажности",
    "робот пылесос": "робот-пылесос",
    "компьютерный монитор": "компьютерный монитор",
    "головные наушники": "наушники",
    "экран для телевизора": "экран для телевизора",
}

DEVICE_SYNONYMS: dict[str, list[str]] = {
    "свет": ["свет", "освещение"],
    "освещение": ["освещение", "свет"],
    "лампа": ["лампа"],
    "светильник": ["светильник"],
    "люстра": ["люстра", "потолочная люстра"],
    "потолочная люстра": ["потолочная люстра", "люстра"],
    "телевизор": ["телевизор", "телик", "телек", "тв", "ящик"],
    "телик": ["телик", "телевизор", "телек", "тв", "ящик"],
    "телек": ["телек", "телик", "телевизор", "тв", "ящик"],
    "тв": ["тв", "телик", "телек", "телевизор", "ящик"],
    "ящик": ["ящик", "телик", "телек", "тв", "телевизор"],
    "саундбар": ["саундбар", "аудиосистема", "звуковая панель", "акустическая система"],
    "аудиосистема": ["аудиосистема", "саундбар"],
    "музыка": ["музыка", "аудиосистема"],
    "плеер": ["плеер", "музыкальный плеер", "проигрыватель музыки", "проигрыватель"],
    "проигрыватель": ["проигрыватель", "плеер"],
    "радио": ["радио", "радиоприемник", "приемник"],
    "радиоприемник": ["радиоприемник", "радио", "приемник"],
    "приемник": ["приемник", "радио"],
    "колонка первая": ["колонка первая", "первая колонка"],
    "колонка вторая": ["колонка вторая", "вторая колонка"],
    "торшер": ["торшер", "напольный светильник"],
    "напольный светильник": ["напольный светильник", "торшер"],
    "светодиодная лента": ["светодиодная лента", "световая лента", "лента", "диодная лента"],
    "световая лента": ["световая лента", "светодиодная лента", "лента"],
    "лента": ["лента", "светодиодная лента"],
    "диодная лента": ["диодная лента", "светодиодная лента"],
    "термометр": ["термометр", "датчик температуры", "градусник"],
    "датчик температуры": ["датчик температуры", "термометр", "градусник"],
    "градусник": ["градусник", "термометр"],
    "гигрометр": ["гигрометр", "датчик влажности"],
    "датчик влажности": ["датчик влажности", "гигрометр"],
    "увлажнитель": ["увлажнитель", "увлажнитель воздуха"],
    "увлажнитель воздуха": ["увлажнитель воздуха", "увлажнитель"],
    "пылесос": ["пылесос", "робот пылесос", "робот"],
    "робот пылесос": ["робот пылесос", "пылесос", "робот"],
    "робот": ["робот", "пылесос"],
    "монитор": ["монитор", "компьютерный монитор"],
    "компьютерный монитор": ["компьютерный монитор", "монитор"],
    "наушники": ["наушники", "головные наушники", "гарнитура", "уши"],
    "головные наушники": ["головные наушники", "наушники", "уши"],
    "гарнитура": ["гарнитура", "наушники"],
    "уши": ["уши", "наушники"],
    "микрофон": ["микрофон", "микро"],
    "микро": ["микро", "микрофон"],
    "кондиционер": ["кондиционер", "кондей"],
    "кондей": ["кондей", "кондиционер"],
}

EDGE_QUERIES = [
    ("я", "свет", "", "light_me"),
    ("Маша", "свет", "", "light_masha"),
    ("Мария", "свет", "", "light_maria"),
    ("Марина", "лампа", "", "lamp_marina"),
    ("Марина", "светильник", "", "lamp_marina"),
    ("Маша", "лампа", "", "light_masha"),
    ("Мария", "лампа", "", "light_maria"),
    ("общий", "свет", "", "light_common"),
    ("общий", "телевизор", "", "tv"),
    ("общий", "телик", "", "tv"),
    ("общий", "экран для телевизора", "", "tv"),
    ("общий", "саундбар", "", "soundbar"),
    ("общий", "акустическая система", "", "soundbar"),
    ("общий", "звуковая панель", "", "soundbar"),
    ("общий", "аудиосистема", "", "soundbar"),
    ("общий", "проектор", "", "projector"),
    ("общий", "плеер", "", "player"),
    ("общий", "музыкальный плеер", "", "player"),
    ("общий", "проигрыватель музыки", "", "player"),
    ("общий", "радио", "", "radio"),
    ("общий", "радиоприемник", "", "radio"),
    ("общий", "первая колонка", "", "speaker_first"),
    ("общий", "вторая колонка", "", "speaker_second"),
    ("общий", "колонка первая", "", "speaker_first"),
    ("общий", "колонка вторая", "", "speaker_second"),
    ("общий", "компьютерный монитор", "", "monitor"),
    ("общий", "головные наушники", "", "headphones"),
    ("общий", "гарнитура", "", "headphones"),
    ("общий", "микрофон", "", "microphone"),
    ("общий", "люстра", "", "chandelier_common"),
    ("Антон", "торшер", "", "floor_lamp_anton"),
    ("Антон", "напольный светильник", "", "floor_lamp_anton"),
    ("Андрей", "световая лента", "", "led_strip_andrey"),
    ("Андрей", "светодиодная лента", "", "led_strip_andrey"),
    ("мама", "термометр", "", "thermometer_mama"),
    ("мама", "датчик температуры", "", "thermometer_mama"),
    ("папа", "гигрометр", "", "hygrometer_papa"),
    ("папа", "датчик влажности", "", "hygrometer_papa"),
    ("общий", "термостат", "", "thermostat_common"),
    ("общий", "вентилятор", "", "fan_common"),
    ("общий", "кондиционер", "", "conditioner_common"),
    ("мама", "увлажнитель воздуха", "", "humidifier_mama"),
    ("папа", "очиститель воздуха", "", "purifier_papa"),
    ("общий", "робот пылесос", "", "vacuum_common"),
    ("общий", "пылесос", "", "vacuum_common"),
    ("Маша", "свет", "спальня", "light_masha_bedroom"),
    ("Маша", "свет", "кухня", "light_masha_kitchen"),
    ("Мария", "свет", "гостиная", "light_maria_living"),
    ("мама", "свет", "спальня", "light_mama_bedroom"),
    ("мама", "свет", "кухня", "light_mama_kitchen"),
    ("я", "свет", "спальня", "light_me_bedroom"),
    ("я", "свет", "кабинет", "light_me_cabinet"),
    ("я", "свет", "кухня", "light_me_kitchen"),
    ("папа", "свет", "кабинет", "light_papa_cabinet"),
    ("Антон", "свет", "спальня", "light_anton_bedroom"),
    ("Андрей", "свет", "балкон", "light_andrey_balcony"),
    ("Артем", "свет", "детская", "light_artem_nursery"),
    ("общий", "свет", "кухня", "light_common_kitchen"),
    ("общий", "свет", "спальня", "light_common_bedroom"),
    ("общий", "свет", "гостиная", "light_common_living"),
    ("общий", "свет", "коридор", "light_common_hallway"),
    ("общий", "свет", "ванная", "light_common_bath"),
    ("общий", "телевизор", "гостиная", "tv_living"),
    ("общий", "телевизор", "кухня", "tv_kitchen"),
    ("общий", "телевизор", "спальня", "tv_bedroom"),
    ("общий", "телик", "гостиная", "tv_living"),
    ("Маша", "телевизор", "спальня", "tv_masha_bedroom"),
    ("Маша", "телик", "спальня", "tv_masha_bedroom"),
    ("общий", "аудиосистема", "гостиная", "soundbar_living"),
    ("общий", "музыка", "кухня", "soundbar_kitchen"),
    ("общий", "саундбар", "кухня", "soundbar_kitchen"),
    ("общий", "плеер", "кухня", "player_kitchen"),
    ("общий", "плеер", "спальня", "player_bedroom"),
    ("общий", "радио", "кухня", "radio_kitchen"),
    ("общий", "первая колонка", "гостиная", "speaker_first_living"),
    ("общий", "вторая колонка", "кухня", "speaker_second_kitchen"),
    ("общий", "проектор", "гостиная", "projector_living"),
    ("я", "монитор", "кабинет", "monitor_me_cabinet"),
    ("общий", "микрофон", "кабинет", "microphone_cabinet"),
    ("Марина", "лампа", "спальня", "lamp_marina_bedroom"),
    ("Антон", "торшер", "гостиная", "floor_lamp_anton_living"),
    ("Андрей", "светодиодная лента", "кабинет", "led_strip_andrey_cabinet"),
    ("мама", "термометр", "кухня", "thermometer_mama_kitchen"),
    ("папа", "гигрометр", "спальня", "hygrometer_papa_bedroom"),
    ("общий", "термостат", "гостиная", "thermostat_living"),
    ("общий", "вентилятор", "гостиная", "fan_living"),
    ("общий", "кондиционер", "спальня", "conditioner_bedroom"),
    ("мама", "увлажнитель", "спальня", "humidifier_mama_bedroom"),
    ("папа", "очиститель воздуха", "гостиная", "purifier_papa_living"),
    ("", "свет", "", "light_common"),
    ("", "свет", "кухня", "light_common_kitchen"),
    ("", "свет", "спальня", "light_common_bedroom"),
    ("", "телевизор", "", "tv"),
    ("", "телевизор", "гостиная", "tv_living"),
    ("", "музыка", "", "soundbar"),
    ("", "музыка", "кухня", "soundbar_kitchen"),
    ("Маша", "лампа", "спальня", "light_masha_bedroom"),
    ("Маша", "лампа", "кухня", "light_masha_kitchen"),
    ("Мария", "лампа", "гостиная", "light_maria_living"),
    ("Марина", "светильник", "спальня", "lamp_marina_bedroom"),
    ("мама", "лампа", "", "light_mama"),
    ("папа", "лампа", "", "light_papa"),
    ("я", "лампа", "", "light_me"),
    ("я", "лампа", "кабинет", "light_me_cabinet"),
    ("Антон", "лампа", "спальня", "light_anton_bedroom"),
    ("общий", "лампа", "ванная", "light_common_bath"),
    ("общий", "лампа", "коридор", "light_common_hallway"),
    ("общий", "освещение", "гостиная", "light_common_living"),
    ("общий", "освещение", "", "light_common"),
    ("Маша", "освещение", "", "light_masha"),
    ("Маша", "освещение", "спальня", "light_masha_bedroom"),
    ("общий", "ящик", "", "tv"),
    ("общий", "телек", "", "tv"),
    ("общий", "тв", "", "tv"),
    ("общий", "ящик", "гостиная", "tv_living"),
    ("общий", "телек", "кухня", "tv_kitchen"),
    ("общий", "тв", "спальня", "tv_bedroom"),
    ("Маша", "ящик", "спальня", "tv_masha_bedroom"),
    ("Маша", "телек", "спальня", "tv_masha_bedroom"),
    ("общий", "кондей", "", "conditioner_common"),
    ("общий", "кондей", "спальня", "conditioner_bedroom"),
    ("общий", "уши", "", "headphones"),
    ("мама", "градусник", "", "thermometer_mama"),
    ("мама", "градусник", "кухня", "thermometer_mama_kitchen"),
    ("общий", "робот", "", "vacuum_common"),
    ("общий", "приемник", "", "radio"),
    ("общий", "проигрыватель", "", "player"),
    ("Андрей", "лента", "", "led_strip_andrey"),
    ("Андрей", "лента", "кабинет", "led_strip_andrey_cabinet"),
    ("общий", "потолочная люстра", "", "chandelier_common"),
    ("общий", "микро", "", "microphone"),
]

PLACE_SYNONYMS = {
    "ванная": ["ванная комната"],
    "кабинет": ["офис"],
}


def unique_devices() -> list[Device]:
    seen: set[str] = set()
    unique: list[Device] = []
    for device in DEVICES:
        if device.id in seen:
            continue
        seen.add(device.id)
        unique.append(device)
    return unique


def device_names(device_id: str) -> list[str]:
    names: list[str] = []
    for device in DEVICES:
        if device.id == device_id and device.name not in names:
            names.append(device.name)
    return names


def make_text(owner: str, device: str, place: str) -> str:
    owner_bit = OWNER_PHRASE.get(owner, f"у {owner}" if owner else "")
    device_bit = DEVICE_PHRASE.get(device, device)
    chunks = ["Включи"]
    if owner_bit:
        chunks.append(owner_bit)
    chunks.append(device_bit)
    if place:
        chunks.append(PLACE_PREP.get(place, f"в {place}"))
    return " ".join(chunks)


def make_case(
    *,
    owner: str,
    device: str,
    place: str,
    device_id: str,
    tag: str,
) -> dict:
    owner_out = "" if owner in {"", "общий"} else owner
    return {
        "id": f"embed:{tag}:{owner}|{device}|{place}|{device_id}",
        "text": make_text(owner, device, place),
        "intent": {
            "device": device,
            "action": "on",
            "owner": owner_out,
            "place": place,
            "value": "",
        },
        "expected": {
            "device_id": device_id,
            "action": "on",
            "owner": owner_out,
            "place": place,
            "value": "",
        },
    }


def build_embed_cases() -> list[dict]:
    cases: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(owner: str, device: str, place: str, device_id: str, tag: str) -> None:
        key = (owner, device, place, device_id)
        if key in seen:
            return
        seen.add(key)
        cases.append(
            make_case(
                owner=owner,
                device=device,
                place=place,
                device_id=device_id,
                tag=tag,
            )
        )

    for device in unique_devices():
        owner = OWNER_NAME[device.owner_id]
        for name in device_names(device.id):
            add(owner, name, device.place, device.id, "canonical")
            for synonym in DEVICE_SYNONYMS.get(name, []):
                add(owner, synonym, device.place, device.id, "synonym")
            for place_syn in PLACE_SYNONYMS.get(device.place, []):
                add(owner, name, place_syn, device.id, "place-syn")

        if device.owner_id == "common" and device.place:
            add("общий", device.name, device.place, device.id, "place-only")
        if device.owner_id == "common" and not device.place:
            add("общий", device.name, "", device.id, "no-place")

    for owner, device, place, device_id in EDGE_QUERIES:
        add(owner, device, place, device_id, "edge")

    cases.sort(key=lambda case: case["id"])
    return cases


PIPELINE_DEVICE_ID = {
    ("light_common", "спальня"): "light_common_bedroom",
    ("light_common", "кухня"): "light_common_kitchen",
    ("light_common", "гостиная"): "light_common_living",
    ("light_masha", "спальня"): "light_masha_bedroom",
    ("light_me", "кухня"): "light_me_kitchen",
    ("light_anton", "спальня"): "light_anton_bedroom",
    ("tv", "гостиная"): "tv_living",
    ("tv", "кухня"): "tv_kitchen",
    ("tv", "спальня"): "tv_bedroom",
    ("soundbar", "кухня"): "soundbar_kitchen",
    ("player", "спальня"): "player_bedroom",
}


def intent_device(device_id: str) -> str:
    if device_id.startswith("light_"):
        return "свет"
    if device_id.startswith("tv"):
        return "телевизор"
    if device_id.startswith("soundbar"):
        return "аудиосистема"
    if device_id.startswith("player"):
        return "плеер"
    return "свет"


EXTRA_PIPELINE_CASES = [
    {
        "text": "Следующая серия",
        "expected": {
            "device_id": "tv",
            "action": "increase",
            "owner": "",
            "place": "",
            "value": "",
        },
    },
    {
        "text": "Предыдущая серия",
        "expected": {
            "device_id": "tv",
            "action": "decrease",
            "owner": "",
            "place": "",
            "value": "",
        },
    },
    {
        "text": "Поставь на паузу",
        "expected": {
            "device_id": "tv",
            "action": "stop",
            "owner": "",
            "place": "",
            "value": "",
        },
    },
    {
        "text": "Включи новости",
        "expected": {
            "device_id": "tv",
            "action": "run_content",
            "owner": "",
            "place": "",
            "value": "новости",
        },
    },
    {
        "text": "Поставь мультики",
        "expected": {
            "device_id": "tv",
            "action": "run_content",
            "owner": "",
            "place": "",
            "value": "мультики",
        },
    },
]


def convert_pipeline_cases(raw: list[dict]) -> list[dict]:
    converted = []
    for index, case in enumerate(raw):
        expected = dict(case["expected"])
        device_id = PIPELINE_DEVICE_ID.get(
            (expected["device_id"], expected["place"]),
            expected["device_id"],
        )
        expected["device_id"] = device_id
        converted.append(
            {
                "id": f"pipeline:{index}:{case['text']}",
                "text": case["text"],
                "intent": {
                    "device": intent_device(device_id),
                    "action": expected["action"],
                    "owner": expected["owner"],
                    "place": expected["place"],
                    "value": expected["value"],
                },
                "expected": expected,
            }
        )
    return converted


def main() -> None:
    embed = build_embed_cases()
    (ROOT / "embed_cases.json").write_text(
        json.dumps(embed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    raw_pipeline = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    if raw_pipeline and "intent" not in raw_pipeline[0]:
        pipeline = convert_pipeline_cases(raw_pipeline)
    else:
        pipeline = convert_pipeline_cases(
            [
                {
                    "text": case["text"],
                    "expected": case["expected"],
                }
                for case in raw_pipeline
            ]
        )

    seen_texts = {case["text"] for case in pipeline}
    extras = convert_pipeline_cases(
        [case for case in EXTRA_PIPELINE_CASES if case["text"] not in seen_texts]
    )
    for offset, case in enumerate(extras):
        case["id"] = f"pipeline:{len(pipeline) + offset}:{case['text']}"
    pipeline.extend(extras)

    (ROOT / "cases.json").write_text(
        json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"embed cases: {len(embed)}")
    print(f"pipeline cases: {len(pipeline)}")


if __name__ == "__main__":
    main()
