from __future__ import annotations

import re

from ollama_runner.inventory.registry import (
    DeviceRegistry,
    area_from_text,
    device_type_from_text,
    fold,
    ordinal_from_mention,
    stem_token,
    token_stems,
)
from ollama_runner.nlu.values import canonical_color
from ollama_runner.semantic import ExplicitSlots, SemanticCommand, Target


_CONTENT_INTENTS = frozenset({
    "content.play",
    "audio.play",
    "video.play",
    "photos.show",
})
_NAVIGATION = frozenset({"media.next", "media.previous"})
_EPISODE = re.compile(
    r"(?<![0-9a-zа-яе])(?:(?:перв|втор|трет|четверт|пят|шест|седьм|восьм|девят|десят)\w+|\d+)\s+сери\w+"
    r"|сери\w+\s+\d+",
    re.IGNORECASE,
)
_RELATIVE = re.compile(r"следующ|предыдущ", re.IGNORECASE)
_MUSIC = re.compile(
    r"(?<![0-9a-zа-яе])(?:альбом\w*|плейлист\w*|песн\w*|трек\w*|исполнител\w*)(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_VIDEO = re.compile(
    r"(?<![0-9a-zа-яе])(?:фильм\w*|кино|сериал\w*|мультик\w*|новост\w*|сезон\w*)(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
# The verb is cut only from the start. «Ведьмака включи у Маши» is a known
# limitation of this backend: the verb stays inside the residual. That must
# not turn into a power command for some other device of the named owner.
_VERB = re.compile(
    r"^(?:пожалуйста\s+)?(?:запусти|включи|выключи|поставь|вруби|покажи|перемотай|сделай|хочу|убавь|увеличь|выруби)\b\s*",
    re.IGNORECASE,
)
_SLOT_PREPOSITION = r"(?:на|в|у|для)\s+"
_DEVICE_NOUN = re.compile(
    r"колонк|ламп|светильник|телевиз|телик|телек|монитор|плеер|проектор|саундбар|микрофон|люстр|торшер"
)
_ORDINALS = (
    ("перв", 1),
    ("втор", 2),
    ("трет", 3),
    ("четвер", 4),
)


def read_slots(text: str, registry: DeviceRegistry) -> ExplicitSlots:
    """Owner, room, ordinal and device type read from the utterance."""

    owner_hit = registry.owner_from_text(text)
    area_hit = area_from_text(text)
    type_hit = device_type_from_text(text)
    if type_hit is None:
        alias_type = registry.unique_alias_type(text)
        if alias_type:
            type_hit = (alias_type, None)
    return ExplicitSlots(
        owner=owner_hit[0] if owner_hit else None,
        raw_owner=owner_hit[1] if owner_hit else None,
        area=area_hit[0] if area_hit else None,
        raw_area=area_hit[1] if area_hit else None,
        ordinal=_device_ordinal(text),
        device_type=type_hit[0] if type_hit else None,
        raw_device_type=type_hit[1] if type_hit else None,
    )


def spoken_mention(text: str, registry: DeviceRegistry) -> str | None:
    """Device words as the user said them, including a neighboring ordinal.

    A longer inventory alias wins over the bare type token, so
    «напольный светильник» stays the whole phrase rather than «светильник».
    """

    hit = device_type_from_text(text)
    typed = _extend_ordinal(text, hit[1]) if hit is not None and hit[1] else None
    alias = _alias_span(text, registry)
    if alias and typed:
        return alias if len(alias) >= len(typed) else typed
    return alias or typed


def apply_explicit(
    text: str,
    command: SemanticCommand,
    registry: DeviceRegistry,
    *,
    authoritative_intent: bool = False,
) -> tuple[SemanticCommand, ExplicitSlots, tuple[str, ...]]:
    slots = read_slots(text, registry)
    ordinal = slots.ordinal

    intent = command.intent
    arguments = dict(command.arguments)
    notes: list[str] = []

    if authoritative_intent:
        notes.append("intent kept from fallback")
    elif _is_episode(text):
        if intent != "video.play":
            notes.append(f"intent {intent} → video.play")
        intent = "video.play"
        arguments["content"] = _episode_span(text)
        arguments.pop("value", None)
    elif _is_relative(text) and intent in _CONTENT_INTENTS | _NAVIGATION:
        direction = "media.previous" if re.search(r"предыдущ", fold(text)) else "media.next"
        if intent != direction:
            notes.append(f"intent {intent} → {direction}")
        intent = direction
        arguments.pop("content", None)
    elif intent in _CONTENT_INTENTS | _NAVIGATION and _content_names_device(arguments.get("content"), registry):
        notes.append(f"intent {intent} → device.turn_on")
        intent = "device.turn_on"
        arguments.pop("content", None)
    elif intent in {"audio.play", "video.play"} and registry.unique_alias_type(text) and _media_kind(text) is None:
        notes.append(f"intent {intent} → device.turn_on")
        intent = "device.turn_on"
        arguments.pop("content", None)
    elif intent == "content.play":
        rewritten = _media_kind(text)
        if rewritten and rewritten != intent:
            notes.append(f"intent {intent} → {rewritten}")
            intent = rewritten
            if not arguments.get("content"):
                arguments["content"] = _leftover(text, registry)
    elif intent == "device.turn_on" and _unknown_leftover(text, registry):
        leftover = _unknown_leftover(text, registry)
        kind = _media_kind(text)
        rewritten = kind or "content.play"
        notes.append(f"intent {intent} → {rewritten}")
        intent = rewritten
        arguments["content"] = arguments.get("content") or leftover
    elif intent in {"device.turn_on", "media.play"} and not registry.alias_mentioned(text):
        rewritten = _media_kind(text)
        if rewritten:
            notes.append(f"intent {intent} → {rewritten}")
            intent = rewritten
            arguments["content"] = arguments.get("content") or _leftover(text, registry)
    elif intent in _NAVIGATION and not _is_relative(text):
        rewritten = _media_kind(text) or "video.play"
        notes.append(f"intent {intent} → {rewritten}")
        intent = rewritten
        arguments["content"] = arguments.get("content") or _leftover(text, registry)

    if intent in {"device.turn_on", "device.turn_off", "media.play", "media.stop"} and arguments.get("value"):
        if registry.alias_mentioned(arguments["value"]) or device_type_from_text(arguments["value"]):
            notes.append("dropped device name from value")
            arguments.pop("value", None)

    if intent == "color.set":
        color = canonical_color(_unconsumed_span(text, registry))
        if color:
            if arguments.get("value") != color:
                notes.append("color from text")
            arguments["value"] = color
        elif not arguments.get("value"):
            raw_color = _unconsumed_span(text, registry)
            if raw_color:
                notes.append("filled color from unconsumed span")
                arguments["value"] = raw_color

    if intent in _CONTENT_INTENTS:
        residual = _residual_content(text, registry)
        if residual:
            if arguments.get("content") != residual:
                notes.append("content from user span")
            arguments["content"] = residual

    owner = slots.owner
    if owner is None:
        owner = registry.canonicalize_owner(command.target.owner)
        if owner and registry.owner_from_text(text) is None:
            owner = None

    area = slots.area
    device_type = slots.device_type
    mention = spoken_mention(text, registry)

    if slots.raw_area and slots.area and fold(slots.raw_area) != fold(slots.area):
        notes.append(f'raw_area="{slots.raw_area}"')
        notes.append(f'canonical_area="{slots.area}"')
    if slots.raw_device_type and slots.device_type and fold(slots.raw_device_type) != fold(slots.device_type):
        notes.append(f'raw_device_type="{slots.raw_device_type}"')
        notes.append(f'canonical_device_type="{slots.device_type}"')
    if slots.raw_owner and slots.owner and fold(slots.raw_owner) != fold(slots.owner):
        notes.append(f'raw_owner="{slots.raw_owner}"')
        notes.append(f'canonical_owner="{slots.owner}"')

    explicit = any((device_type, mention, owner, area, ordinal is not None))
    merged = SemanticCommand(
        intent=intent,
        target=Target(
            device_type=device_type,
            mention=mention,
            owner=owner,
            area=area,
            ordinal=ordinal if ordinal is not None else None,
            explicit=explicit,
            raw_owner=slots.raw_owner,
            raw_area=slots.raw_area,
            raw_device_type=slots.raw_device_type,
        ),
        arguments={key: value for key, value in arguments.items() if value},
    )
    return merged, slots, tuple(notes)


def _extend_ordinal(text: str, raw: str) -> str:
    folded = fold(text)
    needle = fold(raw)
    start = folded.find(needle)
    if start < 0 or _device_ordinal(text) is None:
        return raw if start < 0 else text[start:start + len(needle)]
    end = start + len(needle)
    previous = re.search(r"([0-9a-zа-яе]+)\s+$", folded[:start])
    if previous and any(previous.group(1).startswith(stem) for stem, _number in _ORDINALS):
        start = previous.start(1)
    else:
        following = re.match(r"\s+([0-9a-zа-яе]+)", folded[end:])
        if following and any(following.group(1).startswith(stem) for stem, _number in _ORDINALS):
            end += following.end(1)
    return text[start:end]


def _alias_span(text: str, registry: DeviceRegistry) -> str | None:
    stems = token_stems(text)
    best_alias: str | None = None
    best_size = -1
    for device in registry.devices():
        for alias in device.aliases:
            alias_stems = token_stems(alias)
            if not alias_stems or not alias_stems <= stems:
                continue
            if len(alias_stems) > best_size or (len(alias_stems) == best_size and best_alias and len(alias) > len(best_alias)):
                best_alias = alias
                best_size = len(alias_stems)
    if best_alias is None:
        return None
    wanted = token_stems(best_alias)
    tokens = list(re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+", text))
    window: tuple[int, int, int] | None = None
    for left in range(len(tokens)):
        found: set[str] = set()
        for right in range(left, len(tokens)):
            found.add(stem_token(fold(tokens[right].group())))
            if not wanted <= found:
                continue
            length = right - left
            if window is None or length < window[0]:
                window = (length, tokens[left].start(), tokens[right].end())
            break
    if window is None:
        return None
    return text[window[1]:window[2]]


def _is_episode(text: str) -> bool:
    return _EPISODE.search(fold(text)) is not None


def _is_relative(text: str) -> bool:
    folded = fold(text)
    return _RELATIVE.search(folded) is not None and not _is_episode(text)


def _media_kind(text: str) -> str | None:
    folded = fold(text)
    if _MUSIC.search(folded):
        return "audio.play"
    if _VIDEO.search(folded) or _is_episode(text):
        return "video.play"
    return None


def _unknown_leftover(text: str, registry: DeviceRegistry) -> str | None:
    """Words left after the verb, owner, area and device name.

    «Включи у Маши Sonne» is not a request to turn on some other device of hers.
    """

    leftover = _leftover(text, registry)
    if not leftover:
        return None
    if registry.alias_mentioned(leftover) or device_type_from_text(leftover):
        return None
    if registry.unique_alias_type(text):
        return None
    return leftover


def _content_names_device(content: str | None, registry: DeviceRegistry) -> bool:
    if not content or not content.strip():
        return False
    if _media_kind(content):
        return False
    return registry.alias_mentioned(content)


def _device_ordinal(text: str) -> int | None:
    folded = fold(text)
    for stem, number in _ORDINALS:
        match = re.search(r"(?<![0-9a-zа-яе])" + stem + r"\w*", folded)
        if match is None:
            continue
        window_start = max(0, match.start() - 24)
        window = folded[window_start:match.end() + 24]
        if "сери" in window:
            continue
        if _DEVICE_NOUN.search(window):
            return number
    mention = ordinal_from_mention(text)
    return None if "сери" in folded else None


def _episode_span(text: str) -> str:
    return _residual_content(text, DeviceRegistry.from_static())


def _unconsumed_span(text: str, registry: DeviceRegistry) -> str:
    """Text left after the verb, owner, area and device type.

    Those spans already chose the intent target, so they are not the color.
    The remainder stays raw: «красным» is not rewritten to «красный».
    A later color resolver decides whether this fragment is a supported color.
    """

    span = _leftover(text, registry)
    hit = device_type_from_text(span)
    if hit is not None and hit[1]:
        span = re.sub(re.escape(hit[1]), " ", span, count=1, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", span).strip(" ,.")


def _cut_hit(span: str, hit: tuple[str, str] | None) -> str:
    """Drop an already-found owner or area span, wherever it sits.

    A preposition that only introduces that span goes with it.
    «для» inside «плейлист для уборки» is not an owner or an area, so it stays.
    """

    if hit is None or not hit[1]:
        return span
    pattern = (
        r"(?<![0-9a-zа-яе])(?:"
        + _SLOT_PREPOSITION
        + r")?"
        + re.escape(hit[1])
        + r"(?![0-9a-zа-яе])"
    )
    return re.sub(pattern, " ", span, count=1, flags=re.IGNORECASE)


def _cut_device_ordinal(span: str, text: str) -> str:
    if _device_ordinal(text) is None:
        return span
    folded = fold(span)
    for stem, _number in _ORDINALS:
        match = re.search(r"(?<![0-9a-zа-яе])" + stem + r"\w*", folded)
        if match is None:
            continue
        return span[:match.start()] + " " + span[match.end():]
    return span


def _residual_content(text: str, registry: DeviceRegistry) -> str:
    """User-text fragment left after the verb and explicit slots.

    Owner, area and a device ordinal are removed in any position.
    The fragment keeps the user's letters: «ведьмака» is not «Ведьмака».
    """

    span = _VERB.sub("", text.strip())
    span = _cut_hit(span, registry.owner_from_text(span))
    span = _cut_hit(span, area_from_text(span))
    span = _cut_device_ordinal(span, text)
    return re.sub(r"\s+", " ", span).strip(" ,.")


def _leftover(text: str, registry: DeviceRegistry) -> str:
    span = _VERB.sub("", text.strip())
    owner = registry.owner_from_text(span)
    if owner is not None:
        span = re.sub(re.escape(owner[1]), " ", span, count=1, flags=re.IGNORECASE)
    area = area_from_text(span)
    if area is not None:
        span = re.sub(re.escape(area[1]), " ", span, count=1, flags=re.IGNORECASE)
    span = re.sub(r"\b(?:на|в|у|для)\b", " ", span, flags=re.IGNORECASE)
    span = re.sub(r"\s+", " ", span).strip(" ,.")
    return span
