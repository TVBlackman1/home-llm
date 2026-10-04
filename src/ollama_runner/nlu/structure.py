"""Domain coordination. Not a general Russian parser.

An utterance becomes atomic SemanticCommands before the resolver.
A future classifier, if one is added, should label one clause — not the
whole coordinated utterance. Low confidence on that clause can still fall
through to Ministral. This module does not call either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ollama_runner.inventory.registry import (
    DeviceRegistry,
    area_from_text,
    device_type_from_text,
    fold,
    token_stems,
)
from ollama_runner.nlu.parse import _NEGATION, parse_clause
from ollama_runner.semantic import SemanticCommand, SemanticPlan, Target


_COORD = re.compile(r",|\bи\b|\bа\b", re.IGNORECASE)
_FRAME_VERB = re.compile(
    r"(?<![0-9a-zа-яе])(?:включи|выключи|запусти|поставь|вруби|выруби|сделай|перемотай|"
    r"убавь|увеличь|прибавь|покажи|останови|приостанови|перескочи|хочу)\w*",
    re.IGNORECASE,
)
_CUE = (
    ("media.pause", re.compile(r"(?<![0-9a-zа-яе])(?:пауз\w*|приостанови\w*)", re.IGNORECASE)),
    ("media.stop", re.compile(r"(?<![0-9a-zа-яе])останови(?!сь)", re.IGNORECASE)),
    ("media.next", re.compile(r"(?<![0-9a-zа-яе])(?:перескочи\w*|следующ\w*)", re.IGNORECASE)),
    ("media.previous", re.compile(r"(?<![0-9a-zа-яе])предыдущ\w*", re.IGNORECASE)),
    ("volume.decrease", re.compile(r"(?<![0-9a-zа-яе])потише", re.IGNORECASE)),
    ("volume.increase", re.compile(r"(?<![0-9a-zа-яе])(?:по)?громче", re.IGNORECASE)),
    ("brightness.decrease", re.compile(r"(?<![0-9a-zа-яе])потемнее", re.IGNORECASE)),
    ("brightness.increase", re.compile(r"(?<![0-9a-zа-яе])ярче", re.IGNORECASE)),
    ("device.turn_off", re.compile(r"(?<![0-9a-zа-яе])(?:выключи|выруби)\w*", re.IGNORECASE)),
    ("device.turn_on", re.compile(r"(?<![0-9a-zа-яе])(?:включи|запусти|вруби)\w*", re.IGNORECASE)),
)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    value: str
    raw: str


def parse_plan(text: str, registry: DeviceRegistry) -> SemanticPlan:
    devices = _device_spans(text, registry)
    areas = _area_spans(text)
    owners = _owner_spans(text, registry)
    pieces = _pieces(text) if _coordinated(text, devices, areas, owners) else [text.strip()]
    if len(pieces) == 1 and len(devices) < 2 and len(areas) < 2 and len(owners) < 2:
        return _atomic(text, registry)

    commands: list[SemanticCommand] = []
    unresolved: list[str] = []
    inherited: str | None = None
    for piece in pieces:
        if _NEGATION.search(fold(piece)):
            unresolved.append(piece.strip())
            continue
        built, intent, ok = _expand_piece(piece, registry, inherited)
        if not ok:
            unresolved.append(piece.strip())
            continue
        commands.extend(built)
        if intent:
            inherited = intent

    if unresolved:
        return SemanticPlan(
            commands=tuple(commands),
            fully_parsed=False,
            unresolved_spans=tuple(unresolved),
            reason="partial" if commands else "no_intent_cue",
            evidence=("coordination",),
        )
    if not commands:
        return SemanticPlan(reason="no_intent_cue", unresolved_spans=(text.strip(),))
    return SemanticPlan(
        commands=tuple(commands),
        fully_parsed=True,
        reason="handled",
        evidence=("coordination",),
    )


def _atomic(text: str, registry: DeviceRegistry) -> SemanticPlan:
    parsed = parse_clause(text, registry)
    if not parsed.handled:
        return SemanticPlan(
            fully_parsed=False,
            unresolved_spans=(text.strip(),),
            reason=parsed.reason,
            evidence=parsed.evidence,
        )
    return SemanticPlan(
        commands=(parsed.command(),),
        fully_parsed=True,
        reason="handled",
        evidence=parsed.evidence,
    )


def _separated(spans: list[_Span], text: str) -> bool:
    """Adjacent nouns are one phrase. Coordination needs «и», a comma, or «а» between spans."""

    if len(spans) < 2:
        return False
    folded = fold(text)
    for left, right in zip(spans, spans[1:]):
        if not re.search(r",|\bи\b|\bа\b", folded[left.end:right.start]):
            return False
    return True


def _coordinated(
    text: str,
    devices: list[_Span],
    areas: list[_Span],
    owners: list[_Span],
) -> bool:
    if _separated(devices, text) or _separated(areas, text) or _separated(owners, text):
        return True
    return len(_pieces(text)) > 1


def _pieces(text: str) -> list[str]:
    folded = fold(text)
    coords = list(_COORD.finditer(folded))
    if not coords:
        return [text.strip()]
    pieces: list[str] = []
    start = 0
    for index, coord in enumerate(coords):
        later = coords[index + 1].start() if index + 1 < len(coords) else len(text)
        conjunct = text[coord.end():later]
        # «в гостиной и на кухне выключи свет» is one predicate with a fronted
        # area list. A coordinator opens a new clause only after a predicate
        # has already appeared on the left.
        if _starts_new_clause(conjunct) and _has_predicate(text[start:coord.start()]):
            pieces.append(text[start:coord.start()])
            start = coord.end()
    pieces.append(text[start:])
    return [piece.strip(" ,") for piece in pieces if piece.strip(" ,")]


def _has_predicate(text: str) -> bool:
    """True when this span already contains a verb or a bare command cue."""

    if _local_intent(text):
        return True
    return bool(_FRAME_VERB.search(fold(text)))


def _starts_new_clause(text: str) -> bool:
    """A coordinator opens another command only when the next part has its own predicate or a new verb."""

    if _NEGATION.search(fold(text)) and _has_constituent(text):
        return True
    if _local_intent(text) and _has_constituent(text):
        return True
    if not _FRAME_VERB.search(fold(text)):
        return False
    rest = _FRAME_VERB.sub(" ", text)
    rest = re.sub(r"(?i)\b(?:пожалуйста|ну|уже|мне|чуть)\b", " ", rest)
    return bool(re.search(r"[0-9A-Za-zА-Яа-яЁё]", rest))


def _has_constituent(text: str) -> bool:
    registry = _registry()
    return bool(_device_spans(text, registry) or _area_spans(text) or _owner_spans(text, registry))


def _registry() -> DeviceRegistry:
    return DeviceRegistry.from_static()


def _local_intent(text: str) -> str | None:
    folded = fold(text)
    for intent, pattern in _CUE:
        if pattern.search(folded):
            return intent
    return None


def _expand_piece(
    piece: str,
    registry: DeviceRegistry,
    inherited: str | None,
) -> tuple[list[SemanticCommand], str | None, bool]:
    intent = _local_intent(piece) or inherited
    devices = _device_spans(piece, registry)
    areas = _area_spans(piece)
    owners = _owner_spans(piece, registry)
    if not _separated(devices, piece):
        devices = devices[:1]
    if not _separated(areas, piece):
        areas = areas[:1]
    if not _separated(owners, piece):
        owners = owners[:1]
    if intent is None or not devices:
        parsed = parse_clause(piece, registry)
        if parsed.handled:
            return [parsed.command()], parsed.intent, True
        return [], intent, False

    area_shared, area_local, area_ambiguous = _attach(devices, areas, piece)
    owner_shared, owner_local, owner_ambiguous = _attach(devices, owners, piece)
    if area_ambiguous or owner_ambiguous:
        return [], intent, False
    if len(devices) > 1 and (len(area_shared) > 1 or len(owner_shared) > 1):
        return [], intent, False
    if not devices:
        return [], intent, False

    commands: list[SemanticCommand] = []
    for device in devices:
        device_areas = area_local.get(device, []) or area_shared or [_Span(0, 0, "", "")]
        device_owners = owner_local.get(device, []) or owner_shared or [_Span(0, 0, "", "")]
        if len(device_areas) > 1 and len(device_owners) > 1:
            return [], intent, False
        for area in device_areas:
            for owner in device_owners:
                commands.append(_atomic_command(intent, device, area, owner))
    return commands, intent, True


def _coordinator_between(text: str, left: int, right: int) -> bool:
    return bool(_COORD.search(fold(text[left:right])))


def _attach(
    devices: list[_Span],
    modifiers: list[_Span],
    text: str,
) -> tuple[list[_Span], dict[_Span, list[_Span]], bool]:
    if not modifiers:
        return [], {device: [] for device in devices}, False
    if not devices:
        return modifiers, {}, False
    shared: list[_Span] = []
    local: dict[_Span, list[_Span]] = {device: [] for device in devices}
    first = devices[0].start
    for modifier in modifiers:
        if modifier.end <= first:
            shared.append(modifier)
            continue
        previous = [device for device in devices if device.end <= modifier.start]
        following = [device for device in devices if device.start >= modifier.end]
        if (
            previous
            and following
            and _coordinator_between(text, previous[-1].end, modifier.start)
        ):
            local[following[0]].append(modifier)
            continue
        if not previous:
            shared.append(modifier)
            continue
        local[previous[-1]].append(modifier)
    counts = [len(local[device]) for device in devices]
    ambiguous = len(devices) > 1 and len(modifiers) > 1 and any(count >= 2 for count in counts) and any(count == 0 for count in counts)
    return shared, local, ambiguous


def _atomic_command(intent: str, device: _Span, area: _Span, owner: _Span) -> SemanticCommand:
    target = Target(
        device_type=device.value or None,
        mention=device.raw or None,
        owner=owner.value or None,
        area=area.value or None,
        explicit=True,
        raw_owner=owner.raw or None,
        raw_area=area.raw or None,
        raw_device_type=device.raw or None,
    )
    return SemanticCommand(intent=intent, target=target)


def _area_spans(text: str) -> list[_Span]:
    return _mask_spans(text, _next_area)


def _owner_spans(text: str, registry: DeviceRegistry) -> list[_Span]:
    return _mask_spans(text, lambda current: _next_owner(current, registry))


def _next_area(text: str) -> _Span | None:
    hit = area_from_text(text)
    if hit is None or not hit[1]:
        return None
    start = fold(text).find(fold(hit[1]))
    if start < 0:
        return None
    return _Span(start, start + len(hit[1]), hit[0], text[start:start + len(hit[1])])


def _next_owner(text: str, registry: DeviceRegistry) -> _Span | None:
    hit = registry.owner_from_text(text)
    if hit is None or not hit[1]:
        return None
    start = fold(text).find(fold(hit[1]))
    if start < 0:
        return None
    return _Span(start, start + len(hit[1]), hit[0], text[start:start + len(hit[1])])


def _mask_spans(text: str, finder) -> list[_Span]:
    spans: list[_Span] = []
    masked = list(text)
    for _ in range(8):
        current = "".join(masked)
        found = finder(current)
        if found is None:
            break
        spans.append(_Span(found.start, found.end, found.value, text[found.start:found.end]))
        for index in range(found.start, found.end):
            masked[index] = " "
    spans.sort(key=lambda span: span.start)
    return spans


def _device_spans(text: str, registry: DeviceRegistry) -> list[_Span]:
    tokens = list(re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+", text))
    candidates: list[_Span] = []
    for size in (3, 2, 1):
        for index in range(len(tokens) - size + 1):
            window = tokens[index:index + size]
            raw = text[window[0].start():window[-1].end()]
            found = device_type_from_text(raw)
            if found is not None and found[1] and fold(found[1]) == fold(raw):
                candidates.append(_Span(window[0].start(), window[-1].end(), found[0], raw))
                continue
            stems = token_stems(raw)
            if not stems:
                continue
            types = {
                device.type
                for device in registry.devices()
                for alias in device.aliases
                if token_stems(alias) == stems
            }
            if len(types) == 1:
                candidates.append(_Span(window[0].start(), window[-1].end(), next(iter(types)), raw))
    return _keep_longest(candidates)


def _keep_longest(candidates: list[_Span]) -> list[_Span]:
    chosen: list[_Span] = []
    for span in sorted(candidates, key=lambda item: (-(item.end - item.start), item.start)):
        if any(not (span.end <= other.start or span.start >= other.end) for other in chosen):
            continue
        chosen.append(span)
    chosen.sort(key=lambda item: item.start)
    return chosen

