from __future__ import annotations

from ollama_runner.types import Command, Result


class HomeAssistant:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str = "",
    ) -> None:
        self._url = f"http://{host}:{port}"
        self._token = token

    def send(self, command: Command) -> Result:
        return Result(
            ok=True,
            command=command,
            payload={
                "stub": True,
                "target": self._url,
            },
        )
