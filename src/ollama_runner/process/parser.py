from __future__ import annotations

import json
from pathlib import Path

import httpx

from ollama_runner.types import Intent


INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "device": {
            "type": "string",
        },
        "action": {
            "type": "string",
            "enum": [
                "on",
                "off",
                "run_content",
                "stop",
                "set_state",
                "increase",
                "decrease",
            ],
        },
        "owner": {
            "type": "string",
        },
        "place": {
            "type": "string",
        },
        "value": {
            "type": "string",
        },
    },
    "required": [
        "device",
        "action",
        "owner",
        "place",
        "value",
    ],
    "additionalProperties": False,
}


class OllamaParser:
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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.think = think
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
        )

    def parse(self, text: str) -> Intent:
        # Каждый запрос независимый: system prompt + текущая фраза, без истории.
        response = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                "format": INTENT_SCHEMA,
                "think": self.think,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                },
            },
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return Intent.from_dict(json.loads(content))

    def close(self) -> None:
        self.client.close()
