from ollama_runner.inventory.registry import DeviceRegistry, trailing_index
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.resolve.capability import CapabilityResolver
from tests.test_ha_inventory import _snapshot


def _parsed(text: str):
    return parse_deterministic(text, DeviceRegistry.from_static())


def test_first_lamp_forms_share_ordinal_one():
    phrases = (
        "Выключи лампу 1",
        "Выключи лампу один",
        "Выключи лампу номер один",
        "Выключи первую лампу",
        "Включи первую лампу",
    )
    for text in phrases:
        parsed = _parsed(text)
        assert parsed.handled, text
        assert parsed.intent == "device.turn_off" if text.startswith("Выключи") else "device.turn_on"
        assert parsed.slots.device_type == "light", text
        assert parsed.slots.ordinal == 1, text
        assert parsed.mention and "ламп" in parsed.mention.casefold(), text


def test_second_lamp_forms_share_ordinal_two():
    phrases = (
        "Выключи лампу 2",
        "Выключи лампу два",
        "Выключи лампу номер два",
        "Выключи вторую лампу",
        "Включи вторую лампу",
    )
    for text in phrases:
        parsed = _parsed(text)
        assert parsed.handled, text
        assert parsed.intent == "device.turn_off" if text.startswith("Выключи") else "device.turn_on"
        assert parsed.slots.device_type == "light", text
        assert parsed.slots.ordinal == 2, text
        assert parsed.mention and "ламп" in parsed.mention.casefold(), text


def test_ordinal_words_are_not_commands():
    registry = DeviceRegistry.from_static()
    for text in ("один", "два", "первая", "вторая", "номер один"):
        parsed = parse_deterministic(text, registry)
        assert parsed.intent is None, text
        assert parsed.handled is False
        assert parsed.slots.ordinal is None


def test_existing_ordinal_and_episode_phrases_stay():
    column = _parsed("Включи вторую колонку на кухне")
    assert column.slots.ordinal == 2
    assert column.slots.device_type == "speaker"
    assert column.mention == "вторую колонку"

    episode = _parsed("Запусти маме первую серию Ведьмака")
    assert episode.intent == "video.play"
    assert episode.slots.ordinal is None
    assert "ведьмака" in (episode.content or "").casefold()

    seek = _parsed("Перемотай на пять минут назад")
    assert seek.intent == "media.seek_backward"
    assert seek.slots.ordinal is None

    television = _parsed("Включи телевизор")
    assert television.slots.ordinal is None
    assert television.mention == "телевизор"


def test_trailing_index_ignores_model_numbers():
    assert trailing_index("Лампа 1") == 1
    assert trailing_index("Лампа 2") == 2
    assert trailing_index("Лампа") is None
    assert trailing_index("Samsung The Frame (32) (QE32LS03TBKXRU)") is None


def test_ha_ordinal_selects_the_numbered_bulb():
    inventory = _snapshot()
    resolver = CapabilityResolver(inventory.registry)
    expected = {
        "Выключи лампу 1": "light.bulb_e27_lemon_3",
        "Выключи лампу один": "light.bulb_e27_lemon_3",
        "Выключи лампу номер один": "light.bulb_e27_lemon_3",
        "Выключи первую лампу": "light.bulb_e27_lemon_3",
        "Выключи лампу 2": "light.bulb_e27_lemon_3_2",
        "Выключи лампу два": "light.bulb_e27_lemon_3_2",
        "Выключи лампу номер два": "light.bulb_e27_lemon_3_2",
        "Выключи вторую лампу": "light.bulb_e27_lemon_3_2",
    }
    for text, device_id in expected.items():
        parsed = parse_deterministic(text, inventory.registry)
        resolved = resolver.resolve(parsed.command(), text=text)
        assert resolved.status == "resolved", (text, resolved.status, resolved.reason)
        assert resolved.execution_target_id == device_id, text
