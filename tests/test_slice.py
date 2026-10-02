from __future__ import annotations

import json

import httpx
import pytest

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.backend import LLMNLUBackend
from ollama_runner.nlu.normalize import semantic_command_from_dict
from ollama_runner.nlu.slots import apply_explicit
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import DeviceRuntime, RequestContext, SemanticCommand, Target
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.skills.book import Executor
from ollama_runner.types import Result


class CaptureSink:
    def __init__(self) -> None:
        self.commands = []

    def send(self, command):
        self.commands.append(command)
        return Result(ok=True, command=command)


class MapNLU:
    def __init__(self, commands: dict[str, SemanticCommand]) -> None:
        self._commands = commands

    def parse(self, text: str, context: RequestContext) -> SemanticCommand:
        del context
        return self._commands[text]


def _pipeline(commands: dict[str, SemanticCommand]) -> tuple[SemanticPipeline, CaptureSink, DeviceRegistry]:
    registry = DeviceRegistry.from_static()
    sink = CaptureSink()
    pipeline = SemanticPipeline(
        MapNLU(commands),
        CapabilityResolver(registry),
        Executor(registry),
        registry,
    )
    return pipeline, sink, registry


@pytest.mark.unit
def test_vertical_slice_prints_a_full_trace() -> None:
    pipeline, sink, _ = _pipeline({
        "Поставь на паузу": SemanticCommand(intent="media.pause", target=Target()),
    })
    result = pipeline.run(
        "Поставь на паузу",
        sink,
        state={"tv_living": DeviceRuntime(media="playing")},
    )

    assert result.ok
    assert sink.commands[0].device_id == "tv_living"
    trace = result.payload["trace"]
    assert "intent=media.pause" in trace
    assert "target=unspecified" in trace
    assert "tv_living supports media.pause" in trace
    assert "currently playing" in trace
    assert "SEMANTIC TARGET:\ntv_living" in trace
    assert "EXECUTION TARGET:\ntv_living" in trace
    assert "SKILL:\nMediaSkill" in trace
    assert "HA ACTION:\nmedia_player.media_pause(tv_living)" in trace
    assert result.payload["ha_action"] == "media_player.media_pause(tv_living)"


@pytest.mark.unit
def test_slice_cases_reach_expected_ha_actions() -> None:
    cases = {
        "Включи вторую колонку на кухне": (
            SemanticCommand(
                intent="device.turn_on",
                target=Target(device_type="speaker", mention="вторая колонка", area="кухня", ordinal=2, explicit=True),
            ),
            None,
            "speaker_second_kitchen",
            "speaker_second_kitchen",
            "PowerSkill",
            "homeassistant.turn_on(speaker_second_kitchen)",
        ),
        "Поставь Шрека у Маши": (
            SemanticCommand(
                intent="video.play",
                target=Target(owner="Маша", explicit=True),
                arguments={"content": "Шрека"},
            ),
            None,
            "tv_masha_bedroom",
            "tv_masha_bedroom",
            "MediaSkill",
            'media_player.play_media(tv_masha_bedroom, content="Шрека")',
        ),
        "Следующая серия": (
            SemanticCommand(intent="media.next"),
            {"tv_living": DeviceRuntime(media="playing")},
            "tv_living",
            "tv_living",
            "MediaSkill",
            "media_player.media_next_track(tv_living)",
        ),
        "Сделай телевизор погромче": (
            SemanticCommand(
                intent="volume.increase",
                target=Target(device_type="tv", mention="телевизор", explicit=True),
            ),
            {"tv": DeviceRuntime(media="playing")},
            "tv",
            "soundbar",
            "VolumeSkill",
            "media_player.volume_up(soundbar)",
        ),
        "У Антона в спальне сделай потемнее": (
            SemanticCommand(
                intent="brightness.decrease",
                target=Target(owner="Антон", area="спальня", explicit=True),
            ),
            None,
            "light_anton_bedroom",
            "light_anton_bedroom",
            "LightingSkill",
            "light.brightness_decrease(light_anton_bedroom)",
        ),
        "Включи монитор": (
            SemanticCommand(
                intent="device.turn_on",
                target=Target(device_type="monitor", mention="монитор", explicit=True),
            ),
            None,
            "monitor",
            "monitor",
            "PowerSkill",
            "homeassistant.turn_on(monitor)",
        ),
    }
    pipeline, sink, _ = _pipeline({text: item[0] for text, item in cases.items()})

    for text, (_, state, semantic_id, execution_id, skill, ha_action) in cases.items():
        result = pipeline.run(text, sink, state=state)
        assert result.ok, result.payload["trace"]
        assert result.payload["semantic_target"] == semantic_id
        assert result.payload["execution_target"] == execution_id
        assert result.payload["skill"] == skill
        assert result.payload["ha_action"] == ha_action
        assert text in result.payload["trace"]


@pytest.mark.unit
def test_soundbar_turn_on_is_play_and_idle_pause_is_not_sent() -> None:
    pipeline, sink, _ = _pipeline({
        "Включи саундбар": SemanticCommand(
            intent="device.turn_on",
            target=Target(device_type="soundbar", mention="саундбар", explicit=True),
        ),
        "Поставь на паузу": SemanticCommand(intent="media.pause"),
    })

    turned_on = pipeline.run("Включи саундбар", sink)
    assert turned_on.ok
    assert turned_on.payload["ha_action"] == "media_player.media_play(soundbar)"
    assert sink.commands[-1].action == "on"

    paused = pipeline.run("Поставь на паузу", sink)
    assert not paused.ok
    assert paused.payload["status"] == "clarify"
    assert paused.payload["reason"] == "no_active_device"
    assert len(sink.commands) == 1


@pytest.mark.unit
def test_ambiguous_command_does_not_call_the_sink() -> None:
    pipeline, sink, _ = _pipeline({
        "Включи колонку": SemanticCommand(
            intent="device.turn_on",
            target=Target(device_type="speaker", explicit=True),
        ),
    })
    result = pipeline.run("Включи колонку", sink)

    assert not result.ok
    assert result.error == "ambiguous"
    assert sink.commands == []
    assert set(result.payload["candidates"]) == {"speaker_first", "speaker_second"}


@pytest.mark.unit
def test_missing_content_is_rejected_before_the_sink() -> None:
    pipeline, sink, _ = _pipeline({
        "Поставь": SemanticCommand(
            intent="video.play",
            target=Target(device_type="tv", explicit=True),
        ),
    })
    result = pipeline.run("Поставь", sink)

    assert not result.ok
    assert result.error == "content is missing"
    assert sink.commands == []


@pytest.mark.unit
def test_mention_overrides_a_wrong_device_type() -> None:
    command = semantic_command_from_dict({
        "intent": "device.turn_on",
        "device_type": "soundbar",
        "mention": "вторая колонка",
        "owner": "",
        "area": "кухня",
        "ordinal": 0,
        "explicit": True,
        "content": "",
        "value": "",
    })

    assert command.target.device_type == "speaker"
    assert command.target.ordinal == 2
    assert command.target.area == "кухня"
    assert command.intent == "device.turn_on"


@pytest.mark.unit
def test_numbered_episode_is_video_and_relative_stays_next() -> None:
    registry = DeviceRegistry.from_static()
    numbered = semantic_command_from_dict({
        "intent": "media.next",
        "device_type": "tv",
        "mention": "",
        "owner": "",
        "area": "",
        "ordinal": 5,
        "explicit": False,
        "content": "",
        "value": "",
    })
    merged, slots, _ = apply_explicit(
        "Запусти пятую серию второго сезона Доктора Кто",
        numbered,
        registry,
    )

    assert merged.intent == "video.play"
    assert merged.arguments["content"] == "пятую серию второго сезона Доктора Кто"
    assert merged.target.ordinal is None
    assert slots.owner is None

    relative = semantic_command_from_dict({
        "intent": "media.next",
        "device_type": "",
        "mention": "",
        "owner": "",
        "area": "",
        "ordinal": 0,
        "explicit": False,
        "content": "",
        "value": "",
    })
    kept, _, _ = apply_explicit("Следующая серия", relative, registry)
    assert kept.intent == "media.next"
    assert "content" not in kept.arguments


@pytest.mark.unit
def test_explicit_owner_and_office_come_from_the_text() -> None:
    registry = DeviceRegistry.from_static()
    parsed = semantic_command_from_dict({
        "intent": "device.turn_on",
        "device_type": "monitor",
        "mention": "монитор",
        "owner": "",
        "area": "офис",
        "ordinal": 0,
        "explicit": True,
        "content": "",
        "value": "",
    })
    merged, slots, notes = apply_explicit("Включи мой монитор в офисе", parsed, registry)

    assert slots.owner == "я"
    assert slots.raw_owner == "мой"
    assert merged.target.owner == "я"
    assert merged.target.area == "кабинет"
    assert any('canonical_area="кабинет"' in note for note in notes)


@pytest.mark.unit
def test_unknown_title_is_not_a_power_command() -> None:
    registry = DeviceRegistry.from_static()
    parsed = semantic_command_from_dict({
        "intent": "device.turn_on",
        "device_type": "",
        "mention": "",
        "owner": "Маша",
        "area": "",
        "ordinal": 0,
        "explicit": True,
        "content": "",
        "value": "",
    })
    merged, _, _ = apply_explicit("Включи у Маши Sonne", parsed, registry)
    resolved = CapabilityResolver(registry).resolve(merged, text="Включи у Маши Sonne")

    assert merged.intent == "content.play"
    assert merged.arguments["content"] == "Sonne"
    assert resolved.status == "ambiguous"
    assert resolved.reason == "media_type"
    assert resolved.execution_target_id is None


@pytest.mark.unit
def test_power_value_drops_a_repeated_device_name() -> None:
    registry = DeviceRegistry.from_static()
    parsed = semantic_command_from_dict({
        "intent": "device.turn_on",
        "device_type": "soundbar",
        "mention": "музыка",
        "owner": "",
        "area": "кухня",
        "ordinal": 0,
        "explicit": True,
        "content": "",
        "value": "музыка",
    })
    merged, _, notes = apply_explicit("Включи музыку на кухне", parsed, registry)

    assert merged.intent == "device.turn_on"
    assert "value" not in merged.arguments
    assert "dropped device name from value" in notes


def test_next_episode_is_not_content() -> None:
    command = semantic_command_from_dict({
        "intent": "media.next",
        "device_type": "tv",
        "mention": "",
        "owner": "",
        "area": "",
        "ordinal": 0,
        "explicit": False,
        "content": "следующая серия",
        "value": "",
    })

    assert command.intent == "media.next"
    assert command.target.device_type is None
    assert command.target.explicit is False


@pytest.mark.unit
def test_llm_backend_maps_schema_without_picking_an_entity(tmp_path) -> None:
    prompt = tmp_path / "semantic.md"
    prompt.write_text("semantic", encoding="utf-8")
    payload = {
        "intent": "device.turn_on",
        "device_type": "monitor",
        "mention": "монитор",
        "owner": "",
        "area": "",
        "ordinal": 0,
        "explicit": True,
        "content": "",
        "value": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "ministral-3:3b"
        assert "entity_id" not in json.dumps(body["format"])
        assert body["messages"][1]["content"] == "Включи монитор"
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})

    backend = LLMNLUBackend(
        model="ministral-3:3b",
        prompt_path=prompt,
        base_url="http://ollama.test",
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.test"),
    )
    command = backend.parse("Включи монитор", RequestContext())

    assert command.intent == "device.turn_on"
    assert command.target.device_type == "monitor"
    assert command.target.owner is None
    assert command.target.area is None
