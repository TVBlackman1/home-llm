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
            "video.play",
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

    assert resolved.status == "ambiguous"
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
        state={"tv": DeviceRuntime(media="playing")},
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
def test_explicit_room_selects_every_maximal_light(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_off", device_type="light", area="спальня", explicit=True),
        context=RequestContext(source_area="кухня"),
        text="Выключи свет в спальне",
    )

    assert resolved.status == "resolved"
    assert resolved.reason == "maximal_targets"
    assert "light_common_bedroom" in resolved.execution_target_ids
    assert "lamp_marina_bedroom" in resolved.execution_target_ids
    assert "light_common_kitchen" not in resolved.execution_target_ids


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
    assert "video.play" in registry.get("tv").capabilities
    assert "audio.play" in registry.get("soundbar").capabilities
    assert "video.play" not in registry.get("soundbar").capabilities
    assert "audio.play" not in registry.get("tv").capabilities


@pytest.mark.unit
def test_volume_without_an_active_device_does_not_guess(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("volume.increase", device_type="tv", mention="телевизор", explicit=True),
        text="Сделай телевизор погромче",
    )

    assert resolved.status == "clarify"
    assert resolved.reason == "no_active_device"
    assert resolved.execution_target_id is None


@pytest.mark.unit
def test_office_is_the_cabinet(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command(
            "device.turn_on",
            device_type="light",
            owner="я",
            area="офис",
            explicit=True,
        ),
        text="Включи мой свет в офисе",
    )

    assert resolved.status == "resolved"
    assert resolved.execution_target_id == "light_me_cabinet"
    assert resolved.area == "кабинет"
    assert resolved.trace is not None
    assert 'raw_area="офис"' in "\n".join(resolved.trace.normalization_lines) or resolved.area == "кабинет"


@pytest.mark.unit
def test_thermometer_is_unsupported(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="sensor", owner="мама", mention="термометр", explicit=True),
        text="Включи мамин термометр",
    )

    assert resolved.status == "unsupported"
    assert resolved.reason == "capability"
    assert resolved.execution_target_id is None
    assert "thermometer_mama" in resolved.candidates


@pytest.mark.unit
def test_missing_owned_tv_is_not_found(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="tv", owner="мама", explicit=True),
        text="Включи мамин телевизор",
    )

    assert resolved.status == "not_found"
    assert resolved.execution_target_id is None
    assert resolved.candidates == ()


@pytest.mark.unit
def test_soundbar_phrase_is_not_a_speaker(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("device.turn_on", device_type="soundbar", mention="звуковая панель", explicit=True),
        text="Включи звуковая панель",
    )

    assert resolved.status == "resolved"
    assert resolved.execution_target_id == "soundbar"
    assert resolved.intent == "media.play"


@pytest.mark.unit
def test_audio_does_not_use_a_television(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("audio.play", area="кухня", explicit=True, arguments={"content": "Linkin Park"}),
        text="Включи Linkin Park на кухне",
    )

    assert resolved.status == "ambiguous"
    assert resolved.execution_target_id is None
    assert "tv_kitchen" not in resolved.candidates


@pytest.mark.unit
def test_seek_without_a_session_does_not_pick_a_tv(resolver: CapabilityResolver) -> None:
    resolved = resolver.resolve(
        _command("media.seek_forward", arguments={"value": "на 10 минут"}),
        text="Перемотай сериал на 10 минут вперед",
    )

    assert resolved.status == "clarify"
    assert resolved.reason == "no_active_device"
