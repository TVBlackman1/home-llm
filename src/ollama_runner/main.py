from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


OLLAMA_URL = "http://127.0.0.1:11434"

COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "device": {
            "type": "string",
            "enum": ["light", "tv", "media"],
        },
        "action": {
            "type": "string",
            "enum": [
                "on",
                "off",
                "run",
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


class OllamaRunner:
    def __init__(
        self,
        model: str,
        prompt_path: Path,
        *,
        temperature: float = 0.0,
        keep_alive: str = "30m",
        think: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.think = think

        # Prompt читается с диска один раз и дальше хранится в памяти процесса.
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

        self.client = httpx.Client(
            base_url=OLLAMA_URL,
            timeout=300.0,
        )

    def run(self, text: str) -> dict:
        # Каждый запрос полностью независимый.
        # Никакие предыдущие user/assistant сообщения сюда не добавляются.
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        response = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": messages,
                "format": COMMAND_SCHEMA,
                "think": self.think,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                },
            },
        )

        response.raise_for_status()

        data = response.json()
        content = data["message"]["content"]

        return json.loads(content)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OllamaRunner:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model, e.g. qwen3:4b",
    )

    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="Path to system prompt, e.g. prompts/smart_home.md",
    )

    parser.add_argument(
        "--text",
        help="Single command. If omitted, interactive mode is started.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--keep-alive",
        default="30m",
        help='How long Ollama keeps the model loaded, e.g. "30m", "2h", "-1"',
    )

    args = parser.parse_args()

    with OllamaRunner(
        model=args.model,
        prompt_path=args.prompt,
        temperature=args.temperature,
        keep_alive=args.keep_alive,
    ) as runner:
        if args.text is not None:
            result = runner.run(args.text)

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        # Интерактивный режим.
        # Предыдущие сообщения НЕ сохраняются.
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not text:
                continue

            result = runner.run(text)

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()