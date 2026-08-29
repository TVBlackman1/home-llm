from __future__ import annotations

import pytest

from ollama_runner.pipeline import Pipeline
from ollama_runner.process.cache import PostgresCache
from ollama_runner.settings import get_settings
from ollama_runner.sinks.expect import Expect
from ollama_runner.types import Command, Intent


class FakeParser:
    model = "cache-test"

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, text: str) -> Intent:
        self.calls += 1
        return Intent(device="свет", action="on", owner="Маша")


class FakeResolver:
    def resolve(self, intent: Intent) -> Command:
        return Command(
            device_id="light_masha",
            action=intent.action,
            owner=intent.owner,
        )


def _cache() -> PostgresCache:
    settings = get_settings()
    try:
        return PostgresCache(
            settings.postgres_conninfo(),
            model="pipeline-cache-test",
            inventory_v=1,
        )
    except Exception as exc:
        pytest.skip(f"postgres is not available: {exc}")


def test_postgres_fastpath_roundtrip() -> None:
    cache = _cache()
    parser = FakeParser()
    cache.invalidate("Включи свет Маше")

    pipeline = Pipeline().via(cache).or_else(parser, FakeResolver())

    first = (
        pipeline
        .from_text("Включи свет Маше")
        .into(Expect({"device_id": "light_masha", "action": "on"}))
        .run()
    )
    second = (
        pipeline
        .from_text("включи свет маше")
        .into(Expect({"device_id": "light_masha", "action": "on"}))
        .run()
    )

    cache.close()

    assert first.ok
    assert second.ok
    assert parser.calls == 1
