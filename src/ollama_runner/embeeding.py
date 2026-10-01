from __future__ import annotations

import math
import re

import fasttext
import numpy as np

from ollama_runner.inventory.static import DEVICES, OWNERS


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


def canonical_owner(model, owner_text: str, owners=None) -> str:
    owners = list(OWNERS if owners is None else owners)
    query = owner_text.strip()
    if not query:
        return ""

    scores = owner_scores(model, query, owners=owners)
    best_id = max(scores, key=scores.get)
    return next(owner.name for owner in owners if owner.id == best_id)


def canonical_place(model, place_text: str, devices=None) -> str:
    devices = list(DEVICES if devices is None else devices)
    query = place_text.strip()
    if not query:
        return ""

    places = sorted({device.place for device in devices if device.place})
    if not places:
        return ""

    return max(
        places,
        key=lambda place: (
            1.0
            if exact_match(query, place)
            else text_similarity(model, query, place)
        ),
    )


def owner_scores(model, owner_text: str, owners=None) -> dict[str, float]:
    owners = list(OWNERS if owners is None else owners)

    exact_owner = next(
        (
            owner
            for owner in owners
            if exact_match(owner_text, owner.name)
        ),
        None,
    )

    # Если owner совпал точно, считаем это hard constraint.
    if exact_owner is not None:
        return {
            owner.id: 1.0 if owner.id == exact_owner.id else 0.0
            for owner in owners
        }

    return {
        owner.id: text_similarity(
            model,
            owner_text,
            owner.name,
        )
        for owner in owners
    }


def place_score(model, query_place: str, device_place: str) -> float:
    query = query_place.strip()
    candidate = device_place.strip()

    if not query:
        return 1.0 if not candidate else 0.55

    if not candidate:
        return 0.25

    if exact_match(query, candidate):
        return 1.0

    return text_similarity(model, query, candidate)


def combine_scores(
    device_score: float,
    owner_score: float,
    location_score: float = 1.0,
) -> float:
    return (
        device_score ** 0.50
        * owner_score ** 0.25
        * location_score ** 0.25
    )


def resolve(
    model,
    owner_text: str,
    device_text: str,
    place_text: str = "",
    top_k: int = 3,
    devices=None,
    owners=None,
):
    devices = list(DEVICES if devices is None else devices)
    owners = list(OWNERS if owners is None else owners)
    owner_by_id = {owner.id: owner for owner in owners}
    owner_score_map = owner_scores(model, owner_text, owners=owners)

    results = []

    for device in devices:
        owner = owner_by_id[device.owner_id]
        device_score = text_similarity(model, device_text, device.name)
        owner_score = owner_score_map[owner.id]
        location_score = place_score(model, place_text, device.place)
        score = combine_scores(device_score, owner_score, location_score)
        results.append(
            (
                device,
                owner,
                score,
                device_score,
                owner_score,
                location_score,
            )
        )

    exact_owner = next(
        (owner for owner in owners if exact_match(owner_text, owner.name)),
        None,
    )
    global_best_device = max(results, key=lambda row: row[3])
    global_device_score = global_best_device[3]

    if exact_owner is not None:
        owned = [
            row
            for row in results
            if row[0].owner_id == exact_owner.id
        ]
        if owned:
            best_owned = max(owned, key=lambda row: row[3])
            if best_owned[3] >= global_device_score * 0.8:
                ranked = sorted(owned, key=lambda row: row[2], reverse=True)
            else:
                close = [
                    row
                    for row in results
                    if row[3] >= global_device_score * 0.8
                ]
                ranked = sorted(close, key=lambda row: row[2], reverse=True)
        else:
            ranked = sorted(results, key=lambda row: row[2], reverse=True)
    else:
        ranked = sorted(results, key=lambda row: row[2], reverse=True)

    device_matches = sorted(
        (
            (device, text_similarity(model, device_text, device.name))
            for device in devices
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    owner_matches = sorted(
        (
            (owner, owner_score_map[owner.id])
            for owner in owners
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return (
        ranked[:top_k],
        device_matches,
        owner_matches,
    )


def format_matches(matches) -> str:
    return ", ".join(
        f"{entity.id}:{score:.3f}"
        for entity, score in matches
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="общий")
    parser.add_argument("--device", required=True)
    parser.add_argument("--place", default="")
    args = parser.parse_args()

    model = fasttext.load_model("cc.ru.300.bin")
    results, device_matches, owner_matches = resolve(
        model,
        owner_text=args.owner,
        device_text=args.device,
        place_text=args.place,
    )

    print(f"devices: {format_matches(device_matches)}")
    print(f"owners : {format_matches(owner_matches)}")

    for device, owner, total, device_score, owner_score, location_score in results:
        print(
            f"{device.id}"
            f" place={device.place!r}"
            f" total={total:.3f}"
            f" dev={device_score:.3f}"
            f" owner={owner_score:.3f}"
            f" loc={location_score:.3f}"
            f" ({owner.name})"
        )


if __name__ == "__main__":
    main()