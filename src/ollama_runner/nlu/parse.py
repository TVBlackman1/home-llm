"""Deterministic semantic parse. Ministral runs only when intent stays unknown."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ollama_runner.inventory.registry import DeviceRegistry, device_type_from_text, fold
from ollama_runner.nlu.slots import (
    _is_episode,
    _is_relative,
    _media_kind,
    _residual_content,
    _unknown_leftover,
    read_slots,
    spoken_mention,
)
from ollama_runner.nlu.values import canonical_color, parameter_value
from ollama_runner.semantic import ExplicitSlots, SemanticCommand, Target


_POWER_ON = re.compile(r"^(?:пожалуйста\s+)?(?:включи|запусти|вруби|поставь)\b", re.IGNORECASE)
_POWER_OFF = re.compile(r"^(?:пожалуйста\s+)?(?:выключи|выруби)\b", re.IGNORECASE)
_POWER_ON_WORD = re.compile(
    r"(?<![0-9a-zа-яе])(?:включи|вруби)(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_POWER_OFF_WORD = re.compile(
    r"(?<![0-9a-zа-яе])(?:выключи|выруби)(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_PRETEND_FRAME = re.compile(
    r"(?<![0-9a-zа-яе])(?:сделай\s+вид|представь|допустим|притворись)(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_EXTINGUISH = re.compile(
    r"(?<![0-9a-zа-яе])можно(?:\s+уже)?\s+гасить(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_EXTINGUISH_TYPES = frozenset({"light", "tv"})
_SLEEP_SPEAKER = re.compile(
    r"(?<![0-9a-zа-яе])усыпи(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_PAUSE = re.compile(r"(?<![0-9a-zа-яе])(?:пауз\w*|приостанови\w*)", re.IGNORECASE)
_STOP = re.compile(r"(?<![0-9a-zа-яе])останови(?!сь)", re.IGNORECASE)
_SEEK = re.compile(r"(?<![0-9a-zа-яе])перемотай", re.IGNORECASE)
_REWIND = re.compile(r"(?<![0-9a-zа-яе])отмотай(?![0-9a-zа-яе])", re.IGNORECASE)
_FORWARD = re.compile(r"(?<![0-9a-zа-яе])вперед", re.IGNORECASE)
_BACKWARD = re.compile(r"(?<![0-9a-zа-яе])назад", re.IGNORECASE)
_SKIP = re.compile(r"(?<![0-9a-zа-яе])перескочи", re.IGNORECASE)
_TIME_UNIT = re.compile(
    r"(?<![0-9a-zа-яе])(?:секунд\w*|минут\w*|час(?:а|ов|у|е|ом|ы|ах|ами)?)(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_LISTEN = re.compile(
    r"(?<![0-9a-zа-яе])(?:послушать|слушать)(?![0-9a-zа-яе])",
    re.IGNORECASE,
)
_DARKER = re.compile(r"(?<![0-9a-zа-яе])потемнее", re.IGNORECASE)
_BRIGHTER = re.compile(r"(?<![0-9a-zа-яе])ярче", re.IGNORECASE)
_BRIGHTNESS = re.compile(r"(?<![0-9a-zа-яе])яркост", re.IGNORECASE)
_QUIETER = re.compile(r"(?<![0-9a-zа-яе])потише", re.IGNORECASE)
_LOUDER = re.compile(r"(?<![0-9a-zа-яе])(?:по)?громче", re.IGNORECASE)
_VOLUME = re.compile(r"(?<![0-9a-zа-яе])громкост", re.IGNORECASE)
_DECREASE = re.compile(r"(?<![0-9a-zа-яе])убавь", re.IGNORECASE)
_INCREASE = re.compile(r"(?<![0-9a-zа-яе])(?:увеличь|прибавь)", re.IGNORECASE)
_SET = re.compile(r"(?<![0-9a-zа-яе])поставь", re.IGNORECASE)
_NEGATION = re.compile(r"(?<![0-9a-zа-яе])(?:не|нельзя)(?![0-9a-zа-яе])", re.IGNORECASE)
_PLAY = re.compile(
    r"(?<![0-9a-zа-яе])(?:включи|запусти|вруби|поставь|покажи|хочу)\w*",
    re.IGNORECASE,
)
_FRAME_VERB = re.compile(
    r"(?<![0-9a-zа-яе])(?:включи|выключи|запусти|поставь|вруби|выруби|сделай|перемотай|"
    r"убавь|увеличь|прибавь|покажи|останови|приостанови|перескочи|хочу)\w*",
    re.IGNORECASE,
)
_DOMAIN_TOKEN = re.compile(
    r"пауз\w*|приостанови\w*|перемотай\w*|вперед\w*|назад\w*|следующ\w*|предыдущ\w*|"
    r"перескочи\w*|останови\w*|потемнее|ярче|потише|(?:по)?громче|яркост\w*|громкост\w*|"
    r"убавь\w*|увеличь\w*|прибавь\w*|пожалуйста|чуть|немного|ну|уже|мне|тут|здесь|"
    r"сери\w*|фильм\w*|кино|альбом\w*|плейлист\w*|песн\w*|трек\w*|мультик\w*|"
    r"новост\w*|сезон\w*|исполнител\w*|процент\w*|минут\w*|"
    r"красн\w*|синим|синий|синяя|синее|синие|фиолетов\w*|зелен\w*|оранж\w*|"
    r"теплый|тёплый|бел\w*|перв\w*|втор\w*|трет\w*|четвер\w*|\d+",
    re.IGNORECASE,
)
_FUNCTION_WORD = frozenset({
    "в", "на", "у", "для", "с", "по", "к", "из", "от", "про", "о", "и", "а",
    "что", "нибудь", "что-нибудь", "что-то", "что-либо", "чего-нибудь",
    "ничего", "ничто", "это", "этот", "эта", "эту", "этом", "этой", "эти", "этого",
    "пожалуйста", "мне",
})
_NON_TITLE = ("голов", "точк", "разговор")
_NUMBER_TOKEN = re.compile(
    r"^(?:\d+|один|одна|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|"
    r"одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|"
    r"семнадцать|восемнадцать|девятнадцать|двадцать|тридцать|сорок|пятьдесят|шестьдесят)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeterministicParse:
    """`intent is None` means the grammar declined. An empty slot is known absence."""

    intent: str | None
    slots: ExplicitSlots
    mention: str | None
    content: str | None
    value: str | None
    handled: bool
    evidence: tuple[str, ...]
    reason: str

    def command(self) -> SemanticCommand:
        if self.intent is None:
            raise RuntimeError("deterministic parse has no intent")
        target_bits = (self.slots.device_type, self.mention, self.slots.owner, self.slots.area, self.slots.ordinal is not None)
        arguments: dict[str, str] = {}
        if self.content:
            arguments["content"] = self.content
        if self.value:
            arguments["value"] = self.value
        return SemanticCommand(
            intent=self.intent,
            target=Target(
                device_type=self.slots.device_type,
                mention=self.mention,
                owner=self.slots.owner,
                area=self.slots.area,
                ordinal=self.slots.ordinal,
                explicit=any(target_bits),
                raw_owner=self.slots.raw_owner,
                raw_area=self.slots.raw_area,
                raw_device_type=self.slots.raw_device_type,
            ),
            arguments=arguments,
        )


def parse_clause(text: str, registry: DeviceRegistry) -> DeterministicParse:
    """One clause, no coordination. Callers that see «и» use parse_plan."""

    slots = read_slots(text, registry)
    mention = spoken_mention(text, registry)
    folded = fold(text)
    kind = _media_kind(text)
    residual = _residual_content(text, registry)
    value = _value_for(text, registry)

    decision = _decide(text, folded, registry, kind, residual, value)
    if decision is None:
        return DeterministicParse(
            intent=None,
            slots=slots,
            mention=mention,
            content=None,
            value=value[0] if value else None,
            handled=False,
            evidence=(),
            reason="no_intent_cue",
        )
    intent, content, chosen_value, evidence = decision
    return DeterministicParse(
        intent=intent,
        slots=slots,
        mention=mention,
        content=content,
        value=chosen_value,
        handled=True,
        evidence=evidence,
        reason="handled",
    )


def parse_deterministic(text: str, registry: DeviceRegistry) -> DeterministicParse:
    """Single-command view. A coordinated utterance is not one command."""

    from ollama_runner.nlu.structure import parse_plan

    plan = parse_plan(text, registry)
    if plan.fully_parsed and len(plan.commands) == 1:
        command = plan.commands[0]
        target = command.target
        return DeterministicParse(
            intent=command.intent,
            slots=ExplicitSlots(
                owner=target.owner,
                raw_owner=target.raw_owner,
                area=target.area,
                raw_area=target.raw_area,
                ordinal=target.ordinal,
                device_type=target.device_type,
                raw_device_type=target.raw_device_type,
            ),
            mention=target.mention,
            content=command.arguments.get("content"),
            value=command.arguments.get("value"),
            handled=True,
            evidence=plan.evidence,
            reason="handled",
        )
    return DeterministicParse(
        intent=None,
        slots=read_slots(text, registry),
        mention=spoken_mention(text, registry),
        content=None,
        value=None,
        handled=False,
        evidence=plan.evidence,
        reason=plan.reason or "no_intent_cue",
    )


def _value_for(text: str, registry: DeviceRegistry) -> tuple[str, str] | None:
    from ollama_runner.nlu.slots import _unconsumed_span

    color = canonical_color(_unconsumed_span(text, registry))
    if color:
        return color, "color"
    hit = parameter_value(text)
    if hit is None:
        return None
    return hit.canonical, hit.kind


def _decide(
    text: str,
    folded: str,
    registry: DeviceRegistry,
    kind: str | None,
    residual: str,
    value: tuple[str, str] | None,
) -> tuple[str, str | None, str | None, tuple[str, ...]] | None:
    amount = value[0] if value and value[1] != "color" else None
    color = value[0] if value and value[1] == "color" else None
    if _NEGATION.search(folded):
        return None
    extinguish = _extinguish_off(text, folded, registry)
    if extinguish is not None:
        return extinguish
    sleep_speaker = _sleep_speaker_off(text, folded, registry)
    if sleep_speaker is not None:
        return sleep_speaker
    rewind = _rewind_duration(folded, value)
    if rewind is not None:
        return rewind
    if not _command_frame(text, registry):
        return None

    if _is_episode(text):
        return "video.play", residual or None, None, ("episode",)
    if _PAUSE.search(folded):
        return "media.pause", None, None, ("pause",)
    if _SEEK.search(folded) and _FORWARD.search(folded):
        return "media.seek_forward", None, amount, ("seek_forward",)
    if _SEEK.search(folded) and _BACKWARD.search(folded):
        return "media.seek_backward", None, amount, ("seek_backward",)
    if _SKIP.search(folded) and _TIME_UNIT.search(folded):
        if _FORWARD.search(folded):
            return "media.seek_forward", None, amount, ("seek_forward",)
        if _BACKWARD.search(folded):
            return "media.seek_backward", None, amount, ("seek_backward",)
    if _SKIP.search(folded) or _is_relative(text):
        if _is_relative(text):
            intent = "media.previous" if re.search(r"предыдущ", folded) else "media.next"
        else:
            intent = "media.next"
        return intent, None, None, (intent,)
    if _STOP.search(folded):
        return "media.stop", None, None, ("stop",)
    if color:
        return "color.set", None, color, ("color",)
    if _DARKER.search(folded):
        return "brightness.decrease", None, amount, ("darker",)
    if _BRIGHTER.search(folded):
        return "brightness.increase", None, amount, ("brighter",)
    if _BRIGHTNESS.search(folded) and _DECREASE.search(folded):
        return "brightness.decrease", None, amount, ("brightness_decrease",)
    if _BRIGHTNESS.search(folded) and _INCREASE.search(folded):
        return "brightness.increase", None, amount, ("brightness_increase",)
    if _BRIGHTNESS.search(folded) and (_SET.search(folded) or amount):
        return "brightness.set", None, amount, ("brightness_set",)
    if _QUIETER.search(folded):
        return "volume.decrease", None, amount, ("quieter",)
    if _LOUDER.search(folded):
        return "volume.increase", None, amount, ("louder",)
    if _VOLUME.search(folded) and _DECREASE.search(folded):
        return "volume.decrease", None, amount, ("volume_decrease",)
    if _VOLUME.search(folded) and (_SET.search(folded) or amount):
        return "volume.set", None, amount, ("volume_set",)
    if kind == "audio.play" and _PLAY.search(folded):
        return "audio.play", residual or None, None, ("audio_marker",)
    if kind == "video.play" and _PLAY.search(folded):
        return "video.play", residual or None, None, ("video_marker",)
    if _LISTEN.search(folded) and _POWER_ON.search(text):
        return "audio.play", _without_listen(residual) or None, None, ("listen_infinitive",)
    if _POWER_OFF.search(text) and _known_device(text, registry):
        return "device.turn_off", None, None, ("power_off",)
    if _POWER_ON.search(text) and _known_device(text, registry):
        return "device.turn_on", None, None, ("power_on",)
    if _POWER_ON.search(text) and _unknown_leftover(text, registry) and _plausible_title(residual or _unknown_leftover(text, registry) or ""):
        return "content.play", residual or _unknown_leftover(text, registry), None, ("bare_content",)
    if not _PRETEND_FRAME.search(folded):
        if _POWER_OFF_WORD.search(folded) and _explicit_device(text, registry):
            return "device.turn_off", None, None, ("power_off",)
        if _POWER_ON_WORD.search(folded) and _explicit_device(text, registry):
            return "device.turn_on", None, None, ("power_on",)
    return None


def _without_listen(text: str) -> str:
    cleaned = _LISTEN.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip(" ,.")


def _command_frame(text: str, registry: DeviceRegistry) -> bool:
    """A lone keyword is not a command. A verb, or only domain words, is."""

    if _FRAME_VERB.search(fold(text)):
        return True
    return _domain_only(text, registry)


def _domain_only(text: str, registry: DeviceRegistry) -> bool:
    from ollama_runner.inventory.registry import area_from_text

    span = _DOMAIN_TOKEN.sub(" ", text)
    for _ in range(8):
        owner = registry.owner_from_text(span)
        area = area_from_text(span)
        device = device_type_from_text(span)
        if owner is None and area is None and device is None:
            break
        for hit in (owner, area, device):
            if hit is not None and hit[1]:
                span = re.sub(re.escape(hit[1]), " ", span, count=1, flags=re.IGNORECASE)
    tokens = re.findall(r"[0-9a-zа-яе]+", fold(span))
    return all(token in _FUNCTION_WORD for token in tokens)


def _plausible_title(residual: str) -> bool:
    """A bare leftover is a title only when it is not a number, a pronoun, or a discourse word."""

    tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё-]+", residual)
    if not tokens:
        return False
    folded = [fold(token) for token in tokens]
    return not all(_functionish(token) for token in folded)


def _functionish(token: str) -> bool:
    if token in _FUNCTION_WORD or _NUMBER_TOKEN.match(token):
        return True
    return any(token.startswith(prefix) for prefix in _NON_TITLE)


def _known_device(text: str, registry: DeviceRegistry) -> bool:
    if _unknown_leftover(text, registry):
        return False
    return _explicit_device(text, registry)


def _extinguish_off(
    text: str,
    folded: str,
    registry: DeviceRegistry,
) -> tuple[str, str | None, str | None, tuple[str, ...]] | None:
    """«можно (уже) гасить» plus an explicit light or tv. Not a general power verb."""

    if _PRETEND_FRAME.search(folded) or not _EXTINGUISH.search(folded):
        return None
    found = device_type_from_text(text)
    device_type = found[0] if found is not None else registry.unique_alias_type(text)
    if device_type not in _EXTINGUISH_TYPES:
        return None
    return "device.turn_off", None, None, ("extinguish",)


def _rewind_duration(
    folded: str,
    value: tuple[str, str] | None,
) -> tuple[str, str | None, str | None, tuple[str, ...]] | None:
    """Whole-word «отмотай» plus a parsed duration. The verb supplies backward."""

    if _PRETEND_FRAME.search(folded) or not _REWIND.search(folded):
        return None
    if value is None or value[1] != "duration":
        return None
    if any(re.search(rf"(?<![0-9a-zа-яе]){re.escape(stem)}", folded) for stem in _NON_TITLE):
        return None
    return "media.seek_backward", None, value[0], ("rewind_duration",)


def _sleep_speaker_off(
    text: str,
    folded: str,
    registry: DeviceRegistry,
) -> tuple[str, str | None, str | None, tuple[str, ...]] | None:
    """Whole-word «усыпи» plus an explicit speaker. Not a general power verb."""

    if _PRETEND_FRAME.search(folded) or not _SLEEP_SPEAKER.search(folded):
        return None
    found = device_type_from_text(text)
    device_type = found[0] if found is not None else registry.unique_alias_type(text)
    if device_type != "speaker":
        return None
    return "device.turn_off", None, None, ("sleep_speaker",)


def _explicit_device(text: str, registry: DeviceRegistry) -> bool:
    return (
        device_type_from_text(text) is not None
        or registry.unique_alias_type(text) is not None
    )
