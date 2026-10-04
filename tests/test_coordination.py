from __future__ import annotations

import pytest

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.structure import parse_plan


def _rows(text: str, registry: DeviceRegistry) -> list[tuple[str, str | None, str | None, str | None]]:
    plan = parse_plan(text, registry)
    assert plan.fully_parsed, plan.unresolved_spans
    return [
        (command.intent, command.target.device_type, command.target.area, command.target.owner)
        for command in plan.commands
    ]


@pytest.fixture
def registry() -> DeviceRegistry:
    return DeviceRegistry.from_static()


@pytest.mark.unit
def test_device_coordination(registry: DeviceRegistry) -> None:
    assert _rows("Выключи свет и телевизор", registry) == [
        ("device.turn_off", "light", None, None),
        ("device.turn_off", "tv", None, None),
    ]
    assert _rows("Выключи свет, телевизор", registry) == [
        ("device.turn_off", "light", None, None),
        ("device.turn_off", "tv", None, None),
    ]
    assert _rows("Выключи свет, телевизор и колонку", registry) == [
        ("device.turn_off", "light", None, None),
        ("device.turn_off", "tv", None, None),
        ("device.turn_off", "speaker", None, None),
    ]


@pytest.mark.unit
def test_area_coordination(registry: DeviceRegistry) -> None:
    assert _rows("Выключи свет в гостиной и ванной", registry) == [
        ("device.turn_off", "light", "гостиная", None),
        ("device.turn_off", "light", "ванная", None),
    ]
    assert _rows("Выключи свет на кухне, в ванной и спальне", registry) == [
        ("device.turn_off", "light", "кухня", None),
        ("device.turn_off", "light", "ванная", None),
        ("device.turn_off", "light", "спальня", None),
    ]


@pytest.mark.unit
def test_shared_and_local_modifiers(registry: DeviceRegistry) -> None:
    assert _rows("На кухне выключи свет и телевизор", registry) == [
        ("device.turn_off", "light", "кухня", None),
        ("device.turn_off", "tv", "кухня", None),
    ]
    assert _rows("Выключи свет на кухне и телевизор в спальне", registry) == [
        ("device.turn_off", "light", "кухня", None),
        ("device.turn_off", "tv", "спальня", None),
    ]
    assert _rows("У Маши выключи свет и телевизор", registry) == [
        ("device.turn_off", "light", None, "Маша"),
        ("device.turn_off", "tv", None, "Маша"),
    ]
    assert _rows("Выключи свет у Маши и телевизор у Антона", registry) == [
        ("device.turn_off", "light", None, "Маша"),
        ("device.turn_off", "tv", None, "Антон"),
    ]


@pytest.mark.unit
def test_local_predicate_beats_inherited(registry: DeviceRegistry) -> None:
    assert _rows("Выключи свет и музыку", registry) == [
        ("device.turn_off", "light", None, None),
        ("device.turn_off", "soundbar", None, None),
    ]
    assert _rows("Выруби свет и музыку потише", registry) == [
        ("device.turn_off", "light", None, None),
        ("volume.decrease", "soundbar", None, None),
    ]
    assert _rows("Выключи свет и сделай музыку потише", registry) == [
        ("device.turn_off", "light", None, None),
        ("volume.decrease", "soundbar", None, None),
    ]
    assert _rows("Выключи свет в кухне и ванной, а музыку сделай потише", registry) == [
        ("device.turn_off", "light", "кухня", None),
        ("device.turn_off", "light", "ванная", None),
        ("volume.decrease", "soundbar", None, None),
    ]


@pytest.mark.unit
def test_adjacent_nouns_are_not_coordination(registry: DeviceRegistry) -> None:
    plan = parse_plan("Включи проигрыватель музыки", registry)
    assert plan.fully_parsed
    assert len(plan.commands) == 1
    assert plan.commands[0].intent == "device.turn_on"
    assert plan.commands[0].target.device_type == "player"


@pytest.mark.unit
def test_fronted_areas_share_one_predicate(registry: DeviceRegistry) -> None:
    assert _rows("В гостиной и на кухне выключи свет", registry) == [
        ("device.turn_off", "light", "гостиная", None),
        ("device.turn_off", "light", "кухня", None),
    ]
    assert _rows("В гостиной и на кухне включи свет", registry) == [
        ("device.turn_on", "light", "гостиная", None),
        ("device.turn_on", "light", "кухня", None),
    ]
    assert _rows("В спальне и в кабинете выключи свет", registry) == [
        ("device.turn_off", "light", "спальня", None),
        ("device.turn_off", "light", "кабинет", None),
    ]
    assert _rows("На кухне, в спальне и в гостиной включи свет", registry) == [
        ("device.turn_on", "light", "кухня", None),
        ("device.turn_on", "light", "спальня", None),
        ("device.turn_on", "light", "гостиная", None),
    ]
    assert _rows("в гостиной и кухне выключи свет", registry) == [
        ("device.turn_off", "light", "гостиная", None),
        ("device.turn_off", "light", "кухня", None),
    ]


@pytest.mark.unit
def test_local_branch_does_not_take_the_other_area(registry: DeviceRegistry) -> None:
    assert _rows("выключи свет в гостиной и телевизор на кухне", registry) == [
        ("device.turn_off", "light", "гостиная", None),
        ("device.turn_off", "tv", "кухня", None),
    ]
    assert _rows("в гостиной включи свет, а на кухне телевизор", registry) == [
        ("device.turn_on", "light", "гостиная", None),
        ("device.turn_on", "tv", "кухня", None),
    ]


@pytest.mark.unit
def test_ambiguous_scope_is_not_a_cartesian_product(registry: DeviceRegistry) -> None:
    plan = parse_plan("Выключи свет и телевизор в гостиной и спальне", registry)
    assert not plan.fully_parsed
    assert plan.commands == ()


@pytest.mark.unit
def test_partial_plan_keeps_the_unresolved_span(registry: DeviceRegistry) -> None:
    plan = parse_plan("Выключи свет и сделай Кварис фиолетово-весёлым", registry)
    assert not plan.fully_parsed
    assert [command.intent for command in plan.commands] == ["device.turn_off"]
    assert plan.commands[0].target.device_type == "light"
    assert plan.unresolved_spans


@pytest.mark.unit
def test_negation_is_not_a_positive_command(registry: DeviceRegistry) -> None:
    alone = parse_plan("Не включай телевизор", registry)
    assert not alone.fully_parsed
    assert alone.commands == ()

    compound = parse_plan("Выключи свет и не включай телевизор", registry)
    assert not compound.fully_parsed
    assert [command.intent for command in compound.commands] == ["device.turn_off"]
    assert all(command.target.device_type != "tv" or command.intent != "device.turn_on" for command in compound.commands)
