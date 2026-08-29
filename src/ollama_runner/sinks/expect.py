from __future__ import annotations

import json
from typing import Any

from ollama_runner.types import Command, Result


class Expect:
    def __init__(self, expected: dict[str, Any]) -> None:
        self.expected = expected

    def send(self, command: Command) -> Result:
        actual = command.as_dict()
        diff = {
            key: {
                "expected": value,
                "actual": actual.get(key),
            }
            for key, value in self.expected.items()
            if actual.get(key) != value
        }

        if not diff:
            return Result(ok=True, command=command)

        return Result(
            ok=False,
            command=command,
            error=json.dumps(
                {
                    "diff": diff,
                    "command": actual,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
