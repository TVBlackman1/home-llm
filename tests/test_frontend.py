from __future__ import annotations

import pytest

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.normalize import semantic_command_from_dict
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.nlu.slots import apply_explicit
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import RequestContext
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.skills.book import Executor
from ollama_runner.types import Result


class CaptureSink:
    def __init__(self) -> None:
        self.commands = []

    def send(self, command):
        self.commands.append(command)
        return Result(ok=True, command=command)


def _llm(**overrides):
    payload = {
        "intent": "device.turn_on",
        "device_type": "light",
        "mention": "лампа",
        "owner": "Антон",
        "area": "спальня",
        "ordinal": 4,
        "explicit": True,
        "content": "Ведьмак",
        "value": "антон",
    }
    payload.update(overrides)
    return semantic_command_from_dict(payload)


@pytest.fixture
def registry() -> DeviceRegistry:
    return DeviceRegistry.from_static()


@pytest.mark.unit
def test_obvious_commands_do_not_need_a_model(registry: DeviceRegistry) -> None:
    television = parse_deterministic("Включи телевизор", registry)
    assert television.handled
    assert television.intent == "device.turn_on"
    assert television.slots.device_type == "tv"
    assert television.mention == "телевизор"

    lamp = parse_deterministic("Включи у Антона напольный светильник", registry)
    assert lamp.handled
    assert lamp.mention == "напольный светильник"
    assert lamp.slots.owner == "Антон"

    column = parse_deterministic("Включи вторую колонку на кухне", registry)
    assert column.handled
    assert column.intent == "device.turn_on"
    assert column.slots.device_type == "speaker"
    assert column.slots.ordinal == 2
    assert column.slots.area == "кухня"
    assert column.mention == "вторую колонку"

    nxt = parse_deterministic("Следующая серия", registry)
    assert nxt.handled
    assert nxt.intent == "media.next"
    assert nxt.slots.device_type is None

    title = parse_deterministic("Поставь Интерстеллар", registry)
    assert title.handled
    assert title.intent == "content.play"
    assert title.content == "Интерстеллар"

    shrek = parse_deterministic("Поставь Шрека у Маши", registry)
    assert shrek.intent == "content.play"
    assert shrek.content == "Шрека"
    assert shrek.slots.owner == "Маша"

    typed = parse_deterministic("Поставь песню Sonne", registry)
    assert typed.intent == "audio.play"
    assert typed.content == "песню Sonne"

    vague = parse_deterministic("Что-то здесь слишком громко", registry)
    assert not vague.handled
    assert vague.intent is None
    assert vague.reason == "no_intent_cue"


@pytest.mark.unit
def test_value_keeps_the_spoken_amount_and_canonicalizes_color(registry: DeviceRegistry) -> None:
    brighter = parse_deterministic("Сделай свет немного ярче", registry)
    assert brighter.intent == "brightness.increase"
    assert brighter.value == "немного"

    percent = parse_deterministic("Прибавь у Маши яркость на 10 процентов", registry)
    assert percent.intent == "brightness.increase"
    assert percent.value == "на 10 процентов"
    assert percent.slots.owner == "Маша"

    level = parse_deterministic("Поставь яркость 25 процентов", registry)
    assert level.intent == "brightness.set"
    assert level.value == "25 процентов"

    minutes = parse_deterministic("Перемотай на пять минут назад", registry)
    assert minutes.intent == "media.seek_backward"
    assert minutes.value == "на пять минут"

    color = parse_deterministic("Сделай свет в спальне красным", registry)
    assert color.intent == "color.set"
    assert color.value == "красный"
    assert color.slots.area == "спальня"
    assert color.slots.device_type == "light"


@pytest.mark.unit
def test_deterministic_slots_beat_the_model(registry: DeviceRegistry) -> None:
    text = "У Маши включи вторую колонку на кухне"
    # Verb is not at the start, so the grammar declines and the model is asked.
    declined = parse_deterministic(text, registry)
    assert not declined.handled
    merged, slots, _ = apply_explicit(text, _llm(
        intent="brightness.decrease",
        device_type="light",
        mention="лампа",
        owner="Антон",
        area="спальня",
        ordinal=4,
        content="Колонка",
        value="сильно",
    ), registry)

    assert slots.owner == "Маша"
    assert merged.target.owner == "Маша"
    assert merged.target.area == "кухня"
    assert merged.target.ordinal == 2
    assert merged.target.device_type == "speaker"
    assert merged.target.mention == "вторую колонку"
    assert merged.target.device_type != "light"


@pytest.mark.unit
def test_raw_content_beats_a_normalized_title(registry: DeviceRegistry) -> None:
    text = "Включи ведьмака у Маши"
    parsed = parse_deterministic(text, registry)
    assert parsed.handled
    assert parsed.content == "ведьмака"

    merged, _, _ = apply_explicit(text, _llm(intent="video.play", content="Ведьмак", owner="Антон"), registry)
    assert merged.arguments["content"] == "ведьмака"
    assert merged.target.owner == "Маша"


@pytest.mark.unit
def test_model_cannot_invent_a_target(registry: DeviceRegistry) -> None:
    text = "сделай тут приятнее"
    assert not parse_deterministic(text, registry).handled
    merged, _, _ = apply_explicit(text, _llm(
        intent="brightness.decrease",
        device_type="light",
        mention="лампа",
        owner="Антон",
        area="спальня",
        ordinal=2,
    ), registry)
    assert merged.intent == "brightness.decrease"
    assert merged.target.device_type is None
    assert merged.target.mention is None
    assert merged.target.owner is None
    assert merged.target.area is None
    assert merged.target.ordinal is None
    assert merged.target.explicit is False


class _Boom:
    def parse(self, text: str, context: RequestContext):
        del context
        raise AssertionError(text)


@pytest.mark.unit
def test_bare_title_never_asks_the_model(registry: DeviceRegistry) -> None:
    pipeline = SemanticPipeline(_Boom(), CapabilityResolver(registry), Executor(registry), registry)
    sink = CaptureSink()
    result = pipeline.run("Поставь Шрека у Маши", sink)

    assert pipeline.traces[0].handled
    assert pipeline.traces[0].llm_intent is None
    assert result.payload["status"] == "ambiguous"
    assert result.payload["reason"] == "media_type"
    assert sink.commands == []
    assert result.payload["semantic"]["content"] == "Шрека"


@pytest.mark.unit
def test_handled_brightness_has_no_invented_room(registry: DeviceRegistry) -> None:
    parsed = parse_deterministic("Сделай потемнее", registry)
    assert parsed.handled
    assert parsed.intent == "brightness.decrease"
    assert parsed.slots.device_type is None
    command = parsed.command()
    assert command.target.device_type is None
    assert command.target.explicit is False
