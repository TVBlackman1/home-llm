from __future__ import annotations

from pathlib import Path

import httpx

from ollama_runner.protocols import Transcriber
from ollama_runner.settings import get_settings
from ollama_runner.types import Utterance


class WhisperHttpClient:
    def __init__(
        self,
        base_url: str,
        path: str,
        *,
        timeout: float = 120.0,
    ) -> None:
        self._path = path
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def transcribe(self, audio_path: str) -> str:
        audio = Path(audio_path)
        with audio.open("rb") as handle:
            response = self._client.post(
                self._path,
                files={"file": (audio.name, handle, "application/octet-stream")},
            )
        response.raise_for_status()
        payload = response.json()
        return str(payload["text"])

    def close(self) -> None:
        self._client.close()


class WhisperSource:
    def __init__(
        self,
        audio_path: str | Path,
        *,
        client: Transcriber | None = None,
    ) -> None:
        self._path = Path(audio_path)
        if client is None:
            settings = get_settings()
            client = WhisperHttpClient(
                settings.whisper_url,
                settings.whisper_path,
                timeout=settings.whisper_timeout,
            )
        self._client = client

    def read(self) -> Utterance:
        text = self._client.transcribe(str(self._path)).strip()
        return Utterance(text=text)
