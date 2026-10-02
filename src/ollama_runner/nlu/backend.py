from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import httpx

from ollama_runner.nlu.normalize import semantic_command_from_dict
from ollama_runner.semantic import RequestContext, SemanticCommand


INTENTS = (
    "device.turn_on",
    "device.turn_off",
    "brightness.increase",
    "brightness.decrease",
    "brightness.set",
    "color.set",
    "volume.increase",
    "volume.decrease",
    "volume.set",
    "media.play",
    "media.pause",
    "media.resume",
    "media.stop",
    "media.next",
    "media.previous",
    "media.seek_forward",
    "media.seek_backward",
    "audio.play",
    "video.play",
    "app.launch",
    "screen.share",
    "photos.show",
    "pc.turn_on",
    "pc.turn_off",
)

SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "device_type": {"type": "string"},
        "mention": {"type": "string"},
        "owner": {"type": "string"},
        "area": {"type": "string"},
        "ordinal": {"type": "integer"},
        "explicit": {"type": "boolean"},
        "content": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": [
        "intent",
        "device_type",
        "mention",
        "owner",
        "area",
        "ordinal",
        "explicit",
        "content",
        "value",
    ],
    "additionalProperties": False,
}


class NLUBackend(Protocol):
    """Sync on purpose: the current pipeline, cache and sinks are synchronous.

    A later Hassil or encoder backend returns the same SemanticCommand.
    """

    def parse(self, text: str, context: RequestContext) -> SemanticCommand: ...


class LLMNLUBackend:
    """Current Ollama model, semantic schema. It does not choose a HA entity."""

    def __init__(
        self,
        model: str,
        prompt_path: Path,
        *,
        base_url: str,
        temperature: float = 0.0,
        keep_alive: str = "30m",
        timeout: float = 300.0,
        think: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.think = think
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._owns_client = client is None

    def parse(self, text: str, context: RequestContext | None = None) -> SemanticCommand:
        del context  # reserved for a later prompt section; resolver owns context
        response = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                "format": SEMANTIC_SCHEMA,
                "think": self.think,
                "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature},
            },
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return semantic_command_from_dict(json.loads(content))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
