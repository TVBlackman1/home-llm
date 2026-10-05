"""What home-llm can execute, and which backend performs it.

Advertised device capabilities stay on the registry. They come from
discovery. This module lists only the intents this process can perform.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ollama_runner.ha.normalize import HaExecutionBinding


class ExecutionBackend(Enum):
    """One member today. A later backend is a new member plus an executor."""

    HOME_ASSISTANT = "home_assistant"


@dataclass(frozen=True)
class ExecutionCapability:
    intent: str
    backend: ExecutionBackend


@dataclass(frozen=True)
class ExecutionBinding:
    """Backend plus the target that backend executes against.

    For Home Assistant the target is an entity. It is not a service name
    and not a copy of live state.
    """

    backend: ExecutionBackend
    target: HaExecutionBinding


def _home(*intents: str) -> dict[str, ExecutionCapability]:
    backend = ExecutionBackend.HOME_ASSISTANT
    return {intent: ExecutionCapability(intent, backend) for intent in intents}


EXECUTION_CAPABILITIES = _home(
    "device.turn_on",
    "device.turn_off",
    "brightness.set",
    "brightness.increase",
    "brightness.decrease",
    "color.set",
    "color_temperature.set",
    "color_temperature.warmer",
    "color_temperature.cooler",
    "volume.set",
    "volume.increase",
    "volume.decrease",
    "volume.mute",
    "volume.unmute",
    "source.select",
    "media.play",
    "media.pause",
    "media.resume",
)


def execution_capability(intent: str) -> ExecutionCapability | None:
    """None means home-llm has no implementation, even if a device advertises it."""

    return EXECUTION_CAPABILITIES.get(intent)
