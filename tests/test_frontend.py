from __future__ import annotations

import pytest

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.normalize import semantic_command_from_dict, semantic_outcome_from_dict
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.nlu.slots import apply_explicit
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import (
    CommandOutcome,
    NeedsContextOutcome,
    NotCommandOutcome,
    RequestContext,
    SemanticCommand,
    Target,
)
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
def test_temporal_skip_is_seek_and_item_skip_stays_next(registry: DeviceRegistry) -> None:
    forward = parse_deterministic("Перескочи минуту вперёд", registry)
    assert forward.intent == "media.seek_forward"
    assert forward.value is None

    backward = parse_deterministic("Перескочи минуту назад", registry)
    assert backward.intent == "media.seek_backward"
    assert backward.value is None

    numbered = parse_deterministic("Перескочи две минуты назад", registry)
    assert numbered.intent == "media.seek_backward"
    assert numbered.value == "две минуты"

    seconds = parse_deterministic("Перескочи 30 секунд вперёд", registry)
    assert seconds.intent == "media.seek_forward"
    assert seconds.value is None

    song = parse_deterministic("Перескочи эту песню", registry)
    assert song.intent == "media.next"
    assert song.content is None

    bare = parse_deterministic("Перескочи назад", registry)
    assert bare.intent == "media.next"

    track = parse_deterministic("Следующий трек", registry)
    assert track.intent == "media.next"
    episode = parse_deterministic("Следующая серия", registry)
    assert episode.intent == "media.next"


@pytest.mark.unit
def test_listen_infinitive_is_audio_and_bare_titles_stay_content(registry: DeviceRegistry) -> None:
    elidar = parse_deterministic("Поставь послушать Элидар", registry)
    assert elidar.intent == "audio.play"
    assert elidar.content == "Элидар"

    norven = parse_deterministic("Включи послушать Норвен", registry)
    assert norven.intent == "audio.play"
    assert norven.content == "Норвен"

    kvaris = parse_deterministic("Поставь слушать Кварис", registry)
    assert kvaris.intent == "audio.play"
    assert kvaris.content == "Кварис"

    want = parse_deterministic("Хочу послушать Элидар", registry)
    assert not want.handled
    lets = parse_deterministic("Давай послушаем Элидар", registry)
    assert not lets.handled

    filler = parse_deterministic("слушай, включи свет", registry)
    assert filler.intent != "audio.play"
    assert not filler.handled

    song = parse_deterministic("поставь песню Sonne", registry)
    assert song.intent == "audio.play"
    assert song.content == "песню Sonne"

    bare = parse_deterministic("Поставь Элидар", registry)
    assert bare.intent == "content.play"
    assert bare.content == "Элидар"
    launched = parse_deterministic("Запусти Элидар", registry)
    assert launched.intent == "content.play"
    assert launched.content == "Элидар"
    idiom = parse_deterministic("Запусти двигатель обсуждения", registry)
    assert idiom.intent == "content.play"
    assert idiom.content == "двигатель обсуждения"


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


@pytest.mark.unit
def test_decline_outcomes_drop_a_household_intent() -> None:
    declined = semantic_outcome_from_dict({"outcome": "not_command", "intent": "media.pause"})
    assert isinstance(declined, NotCommandOutcome)
    assert declined.discarded_intent == "media.pause"
    missing = semantic_outcome_from_dict({"outcome": "needs_context"})
    assert isinstance(missing, NeedsContextOutcome)
    assert missing.discarded_intent is None


@pytest.mark.unit
def test_not_command_stops_before_the_resolver(registry: DeviceRegistry) -> None:
    class _Decline:
        def parse(self, text: str, context: RequestContext):
            del text, context
            return NotCommandOutcome()

    pipeline = SemanticPipeline(_Decline(), CapabilityResolver(registry), Executor(registry), registry)
    sink = CaptureSink()
    result = pipeline.run("Пауза в разговоре затянулась", sink)

    assert pipeline.traces[0].handled is False
    assert pipeline.traces[0].llm_intent is None
    assert result.payload["outcome"] == "not_command"
    assert sink.commands == []
    assert pipeline.observations[-1]["final"] is None


def _fallback(registry: DeviceRegistry, text: str, command: SemanticCommand):
    class _Fixed:
        def parse(self, utterance: str, context: RequestContext):
            del utterance, context
            return CommandOutcome(command)

    pipeline = SemanticPipeline(_Fixed(), CapabilityResolver(registry), Executor(registry), registry)
    pipeline.run(text, CaptureSink())
    return pipeline.observations[-1]


@pytest.mark.unit
def test_fallback_command_keeps_llm_intent_and_takes_slots_from_text(registry: DeviceRegistry) -> None:
    text = "У Маши включи вторую колонку на кухне"
    assert not parse_deterministic(text, registry).handled
    observed = _fallback(registry, text, SemanticCommand(
        intent="volume.decrease",
        target=Target(device_type="tv", mention="телевизор", owner="Антон", area="спальня", ordinal=4),
    ))

    assert observed["merge_changed_intent"] is False
    assert observed["llm"]["intent"] == "volume.decrease"
    assert observed["final"]["intent"] == "volume.decrease"
    assert observed["final"]["owner"] == "Маша"
    assert observed["final"]["area"] == "кухня"
    assert observed["final"]["ordinal"] == 2
    assert observed["final"]["device_type"] == "speaker"
    assert observed["final"]["mention"] == "вторую колонку"


@pytest.mark.unit
def test_fallback_command_drops_a_hallucinated_target_and_keeps_intent(registry: DeviceRegistry) -> None:
    text = "сделай тут приятнее"
    assert not parse_deterministic(text, registry).handled
    observed = _fallback(registry, text, SemanticCommand(
        intent="brightness.decrease",
        target=Target(
            device_type="light",
            mention="лампа",
            owner="Антон",
            area="спальня",
            ordinal=2,
            explicit=True,
        ),
    ))

    assert observed["merge_changed_intent"] is False
    assert observed["final"]["intent"] == "brightness.decrease"
    assert observed["final"]["device_type"] is None
    assert observed["final"]["mention"] is None
    assert observed["final"]["owner"] is None
    assert observed["final"]["area"] is None
    assert observed["final"]["ordinal"] is None


@pytest.mark.unit
def test_authoritative_intent_still_canonicalizes_color_and_does_not_reclassify(registry: DeviceRegistry) -> None:
    colored, _, _ = apply_explicit(
        "Сделай свет в спальне красным",
        SemanticCommand(intent="color.set"),
        registry,
        authoritative_intent=True,
    )
    assert colored.intent == "color.set"
    assert colored.arguments["value"] == "красный"
    assert colored.target.area == "спальня"
    assert colored.target.device_type == "light"

    text = "Засвети коридор"
    assert not parse_deterministic(text, registry).handled
    rewritten, _, _ = apply_explicit(text, SemanticCommand(intent="device.turn_on"), registry)
    assert rewritten.intent == "content.play"
    observed = _fallback(registry, text, SemanticCommand(intent="device.turn_on"))
    assert observed["merge_changed_intent"] is False
    assert observed["final"]["intent"] == "device.turn_on"
    assert observed["final"]["area"] == "коридор"


@pytest.mark.unit
def test_deterministic_command_keeps_its_own_intent(registry: DeviceRegistry) -> None:
    class _Boom:
        def parse(self, text: str, context: RequestContext):
            del context
            raise AssertionError(text)

    pipeline = SemanticPipeline(_Boom(), CapabilityResolver(registry), Executor(registry), registry)
    pipeline.run("Сделай потемнее", CaptureSink())
    observed = pipeline.observations[-1]

    assert observed["handled"] is True
    assert observed["outcome"] is None
    assert observed["llm"] is None
    assert observed["final"]["intent"] == "brightness.decrease"
    assert observed["merge_changed_intent"] is False
