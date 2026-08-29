from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from ollama_runner.textnorm import normalize_text
from ollama_runner.types import Command


DDL = """
CREATE TABLE IF NOT EXISTS command_cache (
    model TEXT NOT NULL,
    text_norm TEXT NOT NULL,
    command JSONB NOT NULL,
    inventory_v INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model, text_norm)
);
"""


class MemoryCache:
    def __init__(self) -> None:
        self.store: dict[str, Command] = {}

    def get(self, text: str) -> Command | None:
        return self.store.get(normalize_text(text))

    def put(self, text: str, command: Command) -> None:
        self.store[normalize_text(text)] = command

    def invalidate(self, text: str) -> None:
        self.store.pop(normalize_text(text), None)


class PostgresCache:
    def __init__(
        self,
        conninfo: str,
        *,
        model: str,
        inventory_v: int,
    ) -> None:
        self._model = model
        self._inventory_v = inventory_v
        self._conn = psycopg.connect(conninfo, autocommit=True)
        self._ensure()

    def _ensure(self) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(DDL)

    def get(self, text: str) -> Command | None:
        key = normalize_text(text)
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT command
                FROM command_cache
                WHERE model = %s
                  AND text_norm = %s
                  AND inventory_v = %s
                """,
                (self._model, key, self._inventory_v),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return Command.from_dict(row[0])

    def put(self, text: str, command: Command) -> None:
        key = normalize_text(text)
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO command_cache (model, text_norm, command, inventory_v)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model, text_norm)
                DO UPDATE SET
                    command = EXCLUDED.command,
                    inventory_v = EXCLUDED.inventory_v
                """,
                (
                    self._model,
                    key,
                    Jsonb(command.as_dict()),
                    self._inventory_v,
                ),
            )

    def invalidate(self, text: str) -> None:
        key = normalize_text(text)
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM command_cache
                WHERE model = %s
                  AND text_norm = %s
                """,
                (self._model, key),
            )

    def close(self) -> None:
        self._conn.close()
