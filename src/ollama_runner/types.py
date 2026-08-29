from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Owner:
    id: str
    name: str


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    owner_id: str
    place: str = ""


@dataclass(frozen=True)
class Utterance:
    text: str


@dataclass(frozen=True)
class Intent:
    device: str
    action: str
    owner: str = ""
    place: str = ""
    value: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "device": self.device,
            "action": self.action,
            "owner": self.owner,
            "place": self.place,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Intent:
        return cls(
            device=str(data.get("device", "")),
            action=str(data.get("action", "")),
            owner=str(data.get("owner", "")),
            place=str(data.get("place", "")),
            value=str(data.get("value", "")),
        )


@dataclass(frozen=True)
class Command:
    device_id: str
    action: str
    value: str = ""
    owner: str = ""
    place: str = ""
    kind: str = "device"

    def as_dict(self) -> dict[str, str]:
        return {
            "device_id": self.device_id,
            "action": self.action,
            "value": self.value,
            "owner": self.owner,
            "place": self.place,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Command:
        return cls(
            device_id=str(data["device_id"]),
            action=str(data["action"]),
            value=str(data.get("value", "")),
            owner=str(data.get("owner", "")),
            place=str(data.get("place", "")),
            kind=str(data.get("kind", "device")),
        )


@dataclass(frozen=True)
class Result:
    ok: bool
    command: Command
    payload: dict | None = None
    error: str | None = None
