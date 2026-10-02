from __future__ import annotations

import pytest

from ollama_runner.inventory.registry import (
    DeviceRegistry,
    RegistryDevice,
    capabilities_for,
)
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import DeviceRuntime, RequestContext, SemanticCommand, Target


@pytest.fixture
def resolver() -> CapabilityResolver:
    return CapabilityResolver(DeviceRegistry.from_static())


def _command(intent: str, **target: object) -> SemanticCommand:
    arguments = target.pop("arguments", {})
    return SemanticCommand(
        intent=intent,
        target=Target(**target),
        arguments=arguments,
    )


@pytest.mark.unit
def test_second_speaker_on_kitchen(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "device.turn_on",
            device_type="speaker",
            mention="вторая колонка",
            area="кухня",
            ordinal=2,
            explicit=True,
        ),
        text="Включи вторую колонку на кухне",
    )

    assert resolved.status == "resolved"
    assert resolved.semantic_target_id == "speaker_second_kitchen"
    assert resolved.execution_target_id == "speaker_second_kitchen"


@pytest.mark.unit
def test_second_speaker_without_room_stays_unplaced(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "device.turn_on",
            device_type="speaker",
            ordinal=2,
            explicit=True,
        ),
        text="Включи вторую колонку",
    )

    assert resolved.execution_target_id == "speaker_second"


@pytest.mark.unit
def test_speaker_without_ordinal_is_ambiguous(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="speaker", explicit=True),
        text="Включи колонку",
    )

    assert resolved.status == "ambiguous"
    assert resolved.execution_target_id is None
    assert set(resolved.candidates) == {"speaker_first", "speaker_second"}


@pytest.mark.unit
def test_shrek_at_masha_keeps_raw_content(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "content.play",
            owner="Маша",
            explicit=True,
            arguments={"content": "Шрека"},
        ),
        text="Поставь Шрека у Маши",
    )

    assert resolved.status == "resolved"
    assert resolved.semantic_target_id == "tv_masha_bedroom"
    assert resolved.execution_target_id == "tv_masha_bedroom"
    assert resolved.arguments["content"] == "Шрека"
    assert resolved.owner == "Маша"
    assert resolved.area == ""


@pytest.mark.unit
def test_next_episode_follows_playing_tv(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("media.next"),
        text="Следующая серия",
        state={
            "tv_living": DeviceRuntime(media="playing"),
            "tv_bedroom": DeviceRuntime(media="idle"),
            "speaker_first_living": DeviceRuntime(media="idle"),
        },
    )

    assert resolved.status == "resolved"
    assert resolved.semantic_target_id == "tv_living"
    assert resolved.execution_target_id == "tv_living"
    assert resolved.trace is not None
    assert "currently playing" in "\n".join(resolved.trace.score_lines)


@pytest.mark.unit
def test_pause_without_an_active_device_asks_later(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(_command("media.pause"), text="Поставь на паузу")

    assert resolved.status == "clarify"
    assert resolved.reason == "no_active_device"
    assert resolved.execution_target_id is None
    assert resolved.candidates == ()


@pytest.mark.unit
def test_pause_uses_playing_device(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("media.pause"),
        text="Поставь на паузу",
        state={
            "tv_living": DeviceRuntime(media="playing"),
            "speaker_first_living": DeviceRuntime(media="idle"),
            "soundbar_living": DeviceRuntime(media="idle"),
        },
    )

    assert resolved.status == "resolved"
    assert resolved.reason == "active_device"
    assert resolved.semantic_target_id == "tv_living"
    assert resolved.execution_target_id == "tv_living"


@pytest.mark.unit
def test_pause_with_two_active_devices_asks_later(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("media.pause"),
        state={
            "tv_living": DeviceRuntime(media="playing"),
            "tv_bedroom": DeviceRuntime(media="playing"),
        },
    )

    assert resolved.status == "clarify"
    assert resolved.reason == "multiple_active_devices"
    assert resolved.execution_target_id is None
    assert set(resolved.candidates) == {"tv_living", "tv_bedroom"}


@pytest.mark.unit
def test_next_and_previous_follow_the_single_active_device(resolver: CapabilityResolver) -> None:
    state = {
        "tv_living": DeviceRuntime(media="playing"),
        "tv_kitchen": DeviceRuntime(media="idle"),
    }
    for intent in ("media.next", "media.previous"):
        resolved = resolver.resolve(_command(intent), state=state)
        assert resolved.status == "resolved"
        assert resolved.execution_target_id == "tv_living"


@pytest.mark.unit
def test_next_without_an_active_device_does_not_pick_a_tv(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(_command("media.next"), text="Следующая серия")

    assert resolved.status == "clarify"
    assert resolved.reason == "no_active_device"
    assert resolved.execution_target_id is None


@pytest.mark.unit
def test_resume_uses_the_single_paused_device(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("media.resume"),
        state={
            "tv_bedroom": DeviceRuntime(media="paused"),
            "tv_living": DeviceRuntime(media="idle"),
        },
    )

    assert resolved.execution_target_id == "tv_bedroom"


@pytest.mark.unit
def test_named_speaker_can_be_paused_while_idle(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "media.pause",
            device_type="speaker",
            area="кухня",
            ordinal=2,
            explicit=True,
        )
    )

    assert resolved.status == "resolved"
    assert resolved.execution_target_id == "speaker_second_kitchen"


@pytest.mark.unit
def test_turn_on_without_power_becomes_play(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="soundbar", mention="саундбар", explicit=True),
        text="Включи саундбар",
    )

    assert resolved.status == "resolved"
    assert resolved.intent == "media.play"
    assert resolved.execution_target_id == "soundbar"


@pytest.mark.unit
def test_turn_on_keeps_power_when_the_device_has_it(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="tv", mention="телевизор", explicit=True),
    )

    assert resolved.intent == "device.turn_on"
    assert resolved.execution_target_id == "tv"


@pytest.mark.unit
def test_turn_off_without_power_becomes_stop(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_off", device_type="player", area="кухня", explicit=True),
    )

    assert resolved.intent == "media.stop"
    assert resolved.execution_target_id == "player_kitchen"


@pytest.mark.unit
def test_tv_louder_executes_on_soundbar(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "volume.increase",
            device_type="tv",
            mention="телевизор",
            explicit=True,
        ),
        text="Сделай телевизор погромче",
    )

    assert resolved.status == "resolved"
    assert resolved.semantic_target_id == "tv"
    assert resolved.execution_target_id == "soundbar"
    assert resolved.trace is not None
    assert any("audio_output → soundbar" in line for line in resolved.trace.capability_lines)


@pytest.mark.unit
def test_living_room_tv_louder_uses_living_soundbar(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "volume.increase",
            device_type="tv",
            area="гостиная",
            explicit=True,
        ),
    )

    assert resolved.semantic_target_id == "tv_living"
    assert resolved.execution_target_id == "soundbar_living"


@pytest.mark.unit
def test_anton_bedroom_dims_his_light(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "brightness.decrease",
            owner="Антон",
            area="спальня",
            explicit=True,
        ),
        text="У Антона в спальне сделай потемнее",
    )

    assert resolved.semantic_target_id == "light_anton_bedroom"
    assert resolved.execution_target_id == "light_anton_bedroom"
    assert resolved.owner == "Антон"
    assert resolved.area == "спальня"


@pytest.mark.unit
def test_monitor_does_not_invent_owner(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="monitor", mention="монитор", explicit=True),
        text="Включи монитор",
    )

    assert resolved.status == "resolved"
    assert resolved.semantic_target_id == "monitor"
    assert resolved.execution_target_id == "monitor"
    assert resolved.owner == ""
    assert resolved.area == ""


@pytest.mark.unit
def test_explicit_room_beats_satellite_room(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_off", device_type="light", area="спальня", explicit=True),
        context=RequestContext(source_area="кухня"),
        text="Выключи свет в спальне",
    )

    assert resolved.execution_target_id == "light_common_bedroom"


@pytest.mark.unit
def test_satellite_room_is_a_hint_when_room_is_absent(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_off", device_type="light", explicit=True),
        context=RequestContext(source_area="кухня"),
        text="Выключи свет",
    )

    assert resolved.execution_target_id == "light_common_kitchen"


@pytest.mark.unit
def test_headphones_are_not_a_soundbar(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="headphones", mention="уши", explicit=True),
        text="Включи уши",
    )

    assert resolved.execution_target_id == "headphones"


@pytest.mark.unit
def test_new_device_does_not_need_an_nlu_prompt(resolver: CapabilityResolver) -> None:
    registry = DeviceRegistry.from_static().with_device(
        RegistryDevice(
            id="speaker_third_bath",
            type="speaker",
            owner_id="common",
            area="ванная",
            aliases=("колонка третья",),
            capabilities=capabilities_for("speaker"),
            ordinal=3,
        )
    )
    resolved = CapabilityResolver(registry).resolve(
        _command(
            "device.turn_on",
            device_type="speaker",
            area="ванная",
            ordinal=3,
            explicit=True,
        )
    )

    assert resolved.execution_target_id == "speaker_third_bath"


@pytest.mark.unit
def test_registry_covers_every_static_device() -> None:
    registry = DeviceRegistry.from_static()
    assert registry.get("speaker_second_kitchen") is not None
    assert "media.next" in registry.get("tv_living").capabilities
    assert "volume.increase" not in registry.get("tv").capabilities
    assert "volume.increase" in registry.get("soundbar").capabilities
    assert "content.play" not in registry.get("soundbar").capabilities
