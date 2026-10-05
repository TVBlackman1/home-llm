"""Closed-vocabulary parameters found in the utterance.

Extraction keeps the spoken span («на 10 процентов», «на пять минут»).
Color is the one closed list that also has a canonical form, because the
specification stores the nominative («красным» → «красный»).
"""

from __future__ import annotations

import colorsys
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
_PERCENT_WORDS = {
    "один": 1,
    "одна": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
}
_COLOR_PHRASE = re.compile(r"^(?:теплый|тёплый)\s+бел\w+$", re.IGNORECASE)
_COLOR_WORD = (
    (re.compile(r"^красн\w*$", re.IGNORECASE), "красный"),
    (re.compile(r"^син(?:им|ий|яя|ее|ие)$", re.IGNORECASE), "синий"),
    (re.compile(r"^фиолетов\w*$", re.IGNORECASE), "фиолетовый"),
    (re.compile(r"^зелен\w*$", re.IGNORECASE), "зеленый"),
    (re.compile(r"^оранж\w*$", re.IGNORECASE), "оранжевый"),
)
# Hue 0–360 and saturation 0–100. One table for every color.set name.
# «теплый белый» stays a named color: a warm tint, not a Kelvin command.
_COLOR_HS = {
    "красный": (0, 100),
    "оранжевый": (30, 100),
    "зеленый": (120, 100),
    "синий": (240, 100),
    "фиолетовый": (270, 100),
    "теплый белый": (30, 20),
}


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


def named_color_rgb(name: str) -> tuple[int, int, int] | None:
    """RGB for a canonical color.set name. Unknown names, including «белый», are absent."""

    hs = _COLOR_HS.get(fold(name))
    if hs is None:
        return None
    red, green, blue = colorsys.hsv_to_rgb(hs[0] / 360, hs[1] / 100, 1)
    return (round(red * 255), round(green * 255), round(blue * 255))


_KELVIN = re.compile(
    r"(?<![0-9a-zа-яе])(\d+)\s*кельвин",
    re.IGNORECASE,
)


def kelvin_points(text: str) -> int | None:
    """Kelvin from a bare integer or «4000 кельвинов». Spoken number words are not accepted."""

    if not text:
        return None
    stripped = text.strip()
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    match = _KELVIN.search(fold(stripped))
    if match is None:
        return None
    return int(match.group(1))


def percent_points(value: str | None) -> int | None:
    """Integer read from an already extracted percentage span.

    Digits win. Otherwise one closed word from the same list that recognizes
    the span. «немного» and «сильно» are magnitudes, not percentages.
    """

    if not value:
        return None
    folded = fold(value)
    digit = re.search(r"\d+", folded)
    if digit is not None:
        return int(digit.group())
    for word, number in _PERCENT_WORDS.items():
        if re.search(rf"(?<![0-9a-zа-яе]){word}(?![0-9a-zа-яе])", folded):
            return number
    return None
