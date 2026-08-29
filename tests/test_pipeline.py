from __future__ import annotations

from pathlib import Path

import pytest

from ollama_runner.pipeline import Pipeline
from ollama_runner.process.cache import MemoryCache
from ollama_runner.sinks.expect import Expect
from ollama_runner.types import Command, Intent


class FakeParser:
    model = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def parse(self, text: str) -> Intent:
        self.calls.append(text)
        return Intent(device="свет", action="on", owner="Маша")


class FakeResolver:
    def __init__(self) -> None:
        self.calls: list[Intent] = []

    def resolve(self, intent: Intent) -> Command:
        self.calls.append(intent)
        return Command(
            device_id="light_masha",
            action=intent.action,
            value=intent.value,
            owner=intent.owner,
            place=intent.place,
        )


class FakeTranscriber:
    def transcribe(self, audio_path: str) -> str:
        return Path(audio_path).read_text(encoding="utf-8")


def _pipeline(parser: FakeParser, resolver: FakeResolver, cache: MemoryCache | None = None) -> Pipeline:
    pipeline = Pipeline().or_else(parser, resolver)
    if cache is not None:
        pipeline.via(cache)
    return pipeline


@pytest.mark.unit
def test_text_to_expect() -> None:
    parser = FakeParser()
    resolver = FakeResolver()

    result = (
        _pipeline(parser, resolver)
        .from_text("Включи свет Маше")
        .into(Expect({
            "device_id": "light_masha",
            "action": "on",
            "owner": "Маша",
        }))
        .run()
    )

    assert result.ok
    assert parser.calls == ["Включи свет Маше"]
    assert len(resolver.calls) == 1


@pytest.mark.unit
def test_file_source(tmp_path: Path) -> None:
    path = tmp_path / "cmd.txt"
    path.write_text("Включи свет Маше\n", encoding="utf-8")

    result = (
        _pipeline(FakeParser(), FakeResolver())
        .from_file(path)
        .into(Expect({"device_id": "light_masha", "action": "on"}))
        .run()
    )

    assert result.ok


@pytest.mark.unit
def test_audio_source_uses_transcriber_not_http(tmp_path: Path) -> None:
    path = tmp_path / "cmd.wav"
    path.write_text("Включи свет Маше", encoding="utf-8")

    result = (
        _pipeline(FakeParser(), FakeResolver())
        .from_audio(path, transcriber=FakeTranscriber())
        .into(Expect({"device_id": "light_masha"}))
        .run()
    )

    assert result.ok


@pytest.mark.unit
def test_cache_hit_skips_parser_and_resolver() -> None:
    parser = FakeParser()
    resolver = FakeResolver()
    cache = MemoryCache()

    first = (
        _pipeline(parser, resolver, cache)
        .from_text("Включи свет Маше")
        .into(Expect({"device_id": "light_masha"}))
        .run()
    )
    second = (
        _pipeline(parser, resolver, cache)
        .from_text("включи  свет маше")
        .into(Expect({"device_id": "light_masha"}))
        .run()
    )

    assert first.ok
    assert second.ok
    assert parser.calls == ["Включи свет Маше"]
    assert len(resolver.calls) == 1


@pytest.mark.unit
def test_cache_miss_computes_and_stores() -> None:
    parser = FakeParser()
    resolver = FakeResolver()
    cache = MemoryCache()

    result = (
        _pipeline(parser, resolver, cache)
        .from_text("Включи свет Маше")
        .into(Expect({"device_id": "light_masha"}))
        .run()
    )

    assert result.ok
    stored = cache.get("включи свет маше")
    assert stored is not None
    assert stored.device_id == "light_masha"


@pytest.mark.unit
def test_expect_failure() -> None:
    result = (
        _pipeline(FakeParser(), FakeResolver())
        .from_text("Включи свет Маше")
        .into(Expect({"device_id": "tv"}))
        .run()
    )

    assert not result.ok
    assert result.error is not None
    assert "light_masha" in result.error
