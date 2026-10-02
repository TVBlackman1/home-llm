from __future__ import annotations

from typing import Mapping

from ollama_runner.semantic import DeviceRuntime


# Commands that mean "whatever is playing", not a named device.
SESSION_INTENTS = frozenset({
    "media.pause",
    "media.resume",
    "media.next",
    "media.previous",
    "media.stop",
})


class ActiveSession:
    """The devices that are playing or paused right now.

    A later clarification turn can ask the user when this set is empty
    or has more than one device. This layer does not choose among them.
    """

    def __init__(self, state: Mapping[str, DeviceRuntime]) -> None:
        self._state = state

    def is_active(self, device_id: str, intent: str) -> bool:
        runtime = self._state.get(device_id)
        media = runtime.media if runtime is not None else None
        if intent == "media.resume":
            return media == "paused"
        return media == "playing"
