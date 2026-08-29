from __future__ import annotations

import json

from ollama_runner.types import Command, Result


class PrintSink:
    def send(self, command: Command) -> Result:
        print(
            json.dumps(
                command.as_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return Result(ok=True, command=command)
