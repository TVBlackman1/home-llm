from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ollama_runner.types import Command, Device, Intent, Owner, Result, Utterance


class Source(Protocol):
    def read(self) -> Utterance: ...


class Parser(Protocol):
    model: str

    def parse(self, text: str) -> Intent: ...


class Resolver(Protocol):
    def resolve(self, intent: Intent) -> Command: ...


class Cache(Protocol):
    def get(self, text: str) -> Command | None: ...

    def put(self, text: str, command: Command) -> None: ...

    def invalidate(self, text: str) -> None: ...


class Sink(Protocol):
    def send(self, command: Command) -> Result: ...


class Inventory(Protocol):
    version: int

    def owners(self) -> Sequence[Owner]: ...

    def devices(self) -> Sequence[Device]: ...


class Transcriber(Protocol):
    def transcribe(self, audio_path: str) -> str: ...
