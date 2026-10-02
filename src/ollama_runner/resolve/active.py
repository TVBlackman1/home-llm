from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ollama_runner.semantic import DeviceRuntime


@dataclass(frozen=True)
class IntentPolicy:
    """How an intent uses current playback when the device was not named uniquely."""

    active_media: frozenset[str]
    description: str


_POLICIES = {
    "media.pause": IntentPolicy(frozenset({"playing"}), "media.pause prefers the active media session"),
    "media.stop": IntentPolicy(frozenset({"playing", "paused"}), "media.stop prefers the active media session"),
    "media.next": IntentPolicy(frozenset({"playing"}), "media.next prefers the active media session"),
    "media.previous": IntentPolicy(frozenset({"playing"}), "media.previous prefers the active media session"),
    "media.resume": IntentPolicy(frozenset({"paused"}), "media.resume prefers the paused media session"),
    "media.seek_forward": IntentPolicy(frozenset({"playing"}), "media.seek_forward prefers the active media session"),
    "media.seek_backward": IntentPolicy(frozenset({"playing"}), "media.seek_backward prefers the active media session"),
    "volume.increase": IntentPolicy(frozenset({"playing"}), "volume.increase prefers the active audio session"),
    "volume.decrease": IntentPolicy(frozenset({"playing"}), "volume.decrease prefers the active audio session"),
    "volume.set": IntentPolicy(frozenset({"playing"}), "volume.set prefers the active audio session"),
}


def policy_for(intent: str) -> IntentPolicy | None:
    return _POLICIES.get(intent)


class ActiveSession:
    """Devices whose runtime state satisfies an intent policy.

    Zero or several matches are returned to the caller. This layer does not
    pick one, and it does not invent a device when nothing is active.
    """

    def __init__(self, state: Mapping[str, DeviceRuntime]) -> None:
        self._state = state

    def is_active(self, device_id: str, execution_id: str, policy: IntentPolicy) -> bool:
        for candidate in (device_id, execution_id):
            runtime = self._state.get(candidate)
            media = runtime.media if runtime is not None else None
            if media in policy.active_media:
                return True
        return False
