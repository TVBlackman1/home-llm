from __future__ import annotations

from pathlib import Path

from ollama_runner.types import Utterance


class TextSource:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> Utterance:
        return Utterance(text=self._text)


class FileSource:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read(self) -> Utterance:
        return Utterance(text=self._path.read_text(encoding="utf-8").strip())
