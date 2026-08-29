from __future__ import annotations

import math
import re
from dataclasses import dataclass

import fasttext
import numpy as np


@dataclass(frozen=True)
class Owner:
    id: str
    name: str


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    owner_id: str


OWNERS = [
    Owner("common", "общий"),
    Owner("me", "я"),

    Owner("masha", "Маша"),
    Owner("maria", "Мария"),
    Owner("marina", "Марина"),

    Owner("mama", "мама"),
    Owner("papa", "папа"),

    Owner("anton", "Антон"),
    Owner("andrey", "Андрей"),
]


DEVICES = [
    Device("soundbar", "саундбар", "common"),
    Device("tv", "телевизор", "common"),
    Device("projector", "проектор", "common"),
    Device("player", "плеер", "common"),
    Device("radio", "радио", "common"),
    Device("speaker_first", "колонка первая", "common"),
    Device("speaker_second", "колонка вторая", "common"),

    Device("monitor", "монитор", "common"),
    Device("headphones", "наушники", "common"),
    Device("microphone", "микрофон", "common"),

    Device("light_me", "свет", "me"),
    Device("light_masha", "свет", "masha"),
    Device("light_maria", "свет", "maria"),
    Device("light_common", "свет", "common"),

    Device("lamp_marina", "лампа", "marina"),
    Device("chandelier_common", "люстра", "common"),
    Device("floor_lamp_anton", "торшер", "anton"),
    Device("led_strip_andrey", "светодиодная лента", "andrey"),

    Device("thermometer_mama", "термометр", "mama"),
    Device("hygrometer_papa", "гигрометр", "papa"),
    Device("thermostat_common", "термостат", "common"),
    Device("fan_common", "вентилятор", "common"),
    Device("conditioner_common", "кондиционер", "common"),
    Device("humidifier_mama", "увлажнитель", "mama"),
    Device("purifier_papa", "очиститель воздуха", "papa"),
    Device("vacuum_common", "пылесос", "common"),
]


TESTS = [
    ("я", "свет", "light_me"),
    ("Маша", "свет", "light_masha"),
    ("Мария", "свет", "light_maria"),
    ("общий", "свет", "light_common"),

    ("общий", "телевизор", "tv"),
    ("общий", "саундбар", "soundbar"),
    ("общий", "проектор", "projector"),
    ("общий", "плеер", "player"),
    ("общий", "радио", "radio"),

    ("общий", "первая колонка", "speaker_first"),
    ("общий", "вторая колонка", "speaker_second"),
    ("общий", "колонка первая", "speaker_first"),
    ("общий", "колонка вторая", "speaker_second"),

    ("общий", "телик", "tv"),
    ("общий", "экран для телевизора", "tv"),
    ("общий", "музыкальный плеер", "player"),
    ("общий", "проигрыватель музыки", "player"),
    ("общий", "акустическая система", "soundbar"),
    ("общий", "звуковая панель", "soundbar"),
    ("общий", "радиоприемник", "radio"),

    ("общий", "компьютерный монитор", "monitor"),
    ("общий", "головные наушники", "headphones"),
    ("общий", "гарнитура", "headphones"),
    ("общий", "микрофон", "microphone"),

    ("Марина", "светильник", "lamp_marina"),
    ("Марина", "лампа", "lamp_marina"),
    ("общий", "люстра", "chandelier_common"),
    ("Антон", "напольный светильник", "floor_lamp_anton"),
    ("Антон", "торшер", "floor_lamp_anton"),
    ("Андрей", "световая лента", "led_strip_andrey"),
    ("Андрей", "светодиодная лента", "led_strip_andrey"),

    ("мама", "термометр", "thermometer_mama"),
    ("мама", "датчик температуры", "thermometer_mama"),
    ("папа", "гигрометр", "hygrometer_papa"),
    ("папа", "датчик влажности", "hygrometer_papa"),
    ("общий", "термостат", "thermostat_common"),

    ("общий", "вентилятор", "fan_common"),
    ("общий", "кондиционер", "conditioner_common"),
    ("мама", "увлажнитель воздуха", "humidifier_mama"),
    ("папа", "очиститель воздуха", "purifier_papa"),

    ("Маша", "лампа", "light_masha"),
    ("Мария", "лампа", "light_maria"),
    ("Марина", "лампа", "lamp_marina"),

    ("общий", "робот пылесос", "vacuum_common"),
    ("общий", "пылесос", "vacuum_common"),
]


OWNER_BY_ID = {
    owner.id: owner
    for owner in OWNERS
}


def similarity(model, a: str, b: str) -> float:
    va = model.get_word_vector(a)
    vb = model.get_word_vector(b)

    denom = np.linalg.norm(va) * np.linalg.norm(vb)

    if denom == 0:
        return 0.0

    return float(np.dot(va, vb) / denom)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return math.exp(
        sum(math.log(max(v, 1e-9)) for v in values)
        / len(values)
    )


def text_similarity(model, query: str, candidate: str) -> float:
    query_tokens = tokenize(query)
    candidate_tokens = tokenize(candidate)

    if not query_tokens or not candidate_tokens:
        return 0.0

    scores = []

    for candidate_token in candidate_tokens:
        best = max(
            similarity(model, candidate_token, query_token)
            for query_token in query_tokens
        )

        scores.append(max(best, 0.0))

    return geometric_mean(scores)


def exact_match(query: str, candidate: str) -> bool:
    return query.strip().casefold() == candidate.strip().casefold()


def owner_scores(model, owner_text: str) -> dict[str, float]:
    exact_owner = next(
        (
            owner
            for owner in OWNERS
            if exact_match(owner_text, owner.name)
        ),
        None,
    )

    # Если owner совпал точно, считаем это hard constraint.
    if exact_owner is not None:
        return {
            owner.id: 1.0 if owner.id == exact_owner.id else 0.0
            for owner in OWNERS
        }

    return {
        owner.id: text_similarity(
            model,
            owner_text,
            owner.name,
        )
        for owner in OWNERS
    }


def combine_scores(
    device_score: float,
    owner_score: float,
) -> float:
    return (
        device_score ** 0.65
        * owner_score ** 0.35
    )


def resolve(
    model,
    owner_text: str,
    device_text: str,
    top_k: int = 3,
):
    owners = owner_scores(model, owner_text)

    results = []

    # ВАЖНО:
    # считаем score для КАЖДОГО реально существующего устройства.
    # Никакого предварительного top-3 по device/owner.
    for device in DEVICES:
        owner = OWNER_BY_ID[device.owner_id]

        device_score = text_similarity(
            model,
            device_text,
            device.name,
        )

        owner_score = owners[owner.id]

        score = combine_scores(
            device_score,
            owner_score,
        )

        results.append(
            (
                device,
                owner,
                score,
                device_score,
                owner_score,
            )
        )

    results.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    device_matches = sorted(
        (
            (
                device,
                text_similarity(
                    model,
                    device_text,
                    device.name,
                ),
            )
            for device in DEVICES
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    owner_matches = sorted(
        (
            (
                owner,
                owners[owner.id],
            )
            for owner in OWNERS
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return (
        results[:top_k],
        device_matches,
        owner_matches,
    )


def format_matches(matches) -> str:
    return ", ".join(
        f"{entity.id}:{score:.3f}"
        for entity, score in matches
    )


def main() -> None:
    model = fasttext.load_model("cc.ru.300.bin")

    passed = 0

    for owner_text, device_text, expected in TESTS:
        results, device_matches, owner_matches = resolve(
            model,
            owner_text=owner_text,
            device_text=device_text,
        )

        actual = results[0][0].id if results else None
        score = results[0][2] if results else 0.0

        ok = actual == expected
        passed += ok

        if ok:
            print(
                f"PASS | "
                f"owner={owner_text!r} "
                f"device={device_text!r} "
                f"-> {actual} "
                f"score={score:.3f}"
            )
            continue

        print(
            f"FAIL | "
            f"owner={owner_text!r} "
            f"device={device_text!r} | "
            f"expected={expected} "
            f"actual={actual} "
            f"score={score:.3f}"
        )

        print(
            f"     devices: {format_matches(device_matches)}"
        )

        print(
            f"     owners : {format_matches(owner_matches)}"
        )

        if results:
            print(
                "     final  : "
                + ", ".join(
                    (
                        f"{device.id}"
                        f"(total={total:.3f},"
                        f" dev={device_score:.3f},"
                        f" owner={owner_score:.3f})"
                    )
                    for (
                        device,
                        owner,
                        total,
                        device_score,
                        owner_score,
                    ) in results
                )
            )

    print()
    print(f"{passed}/{len(TESTS)} passed")


if __name__ == "__main__":
    main()