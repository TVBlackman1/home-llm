"""Closed-vocabulary parameters found in the utterance.

Extraction keeps the spoken span («на 10 процентов», «на пять минут»).
Color is the one closed list that also has a canonical form, because the
specification stores the nominative («красным» → «красный»).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ollama_runner.inventory.registry import fold


_NUMBER = (
    r"\d+|один|одна|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|"
    r"одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|"
    r"семнадцать|восемнадцать|девятнадцать|двадцать|тридцать|сорок|пятьдесят|шестьдесят"
)
_PERCENT = re.compile(
    rf"((?:на\s+)?(?:{_NUMBER})\s*(?:процент\w*|%))",
    re.IGNORECASE,
)
_DURATION = re.compile(
    rf"((?:на\s+)?(?<![0-9a-zа-яе])(?:{_NUMBER}|пару)(?![0-9a-zа-яе])\s+минут\w*)",
    re.IGNORECASE,
)
_MAGNITUDE = re.compile(r"(?<![0-9a-zа-яе])(немного|сильно)(?![0-9a-zа-яе])", re.IGNORECASE)
_COLOR_PHRASE = re.compile(r"^(?:теплый|тёплый)\s+бел\w+$", re.IGNORECASE)
_COLOR_WORD = (
    (re.compile(r"^красн\w*$", re.IGNORECASE), "красный"),
    (re.compile(r"^син(?:им|ий|яя|ее|ие)$", re.IGNORECASE), "синий"),
    (re.compile(r"^фиолетов\w*$", re.IGNORECASE), "фиолетовый"),
    (re.compile(r"^зелен\w*$", re.IGNORECASE), "зеленый"),
    (re.compile(r"^оранж\w*$", re.IGNORECASE), "оранжевый"),
)


@dataclass(frozen=True)
class ValueHit:
    raw: str
    canonical: str
    kind: str


def parameter_value(text: str) -> ValueHit | None:
    """Percentage, duration, or magnitude, in that order. Color is separate."""

    percent = _PERCENT.search(text)
    if percent is not None:
        raw = re.sub(r"\s+", " ", percent.group(1)).strip()
        return ValueHit(raw, raw, "percentage")
    duration = _DURATION.search(text)
    if duration is not None:
        raw = re.sub(r"\s+", " ", duration.group(1)).strip()
        return ValueHit(raw, raw, "duration")
    magnitude = _MAGNITUDE.search(fold(text))
    if magnitude is not None:
        raw = text[magnitude.start(1):magnitude.end(1)]
        return ValueHit(raw, fold(raw), "magnitude")
    return None


def canonical_color(span: str) -> str | None:
    """Nominative color when the whole span is one supported color name."""

    text = re.sub(r"\s+", " ", span).strip(" ,.")
    if not text:
        return None
    if _COLOR_PHRASE.match(text):
        return "теплый белый"
    for pattern, canonical in _COLOR_WORD:
        if pattern.match(text):
            return canonical
    return None
