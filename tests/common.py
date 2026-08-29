from __future__ import annotations

import atexit
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from ollama_runner.factory import close_resources, default_pipeline, get_resolver
from ollama_runner.pipeline import Pipeline
from ollama_runner.sinks.expect import Expect
from ollama_runner.types import Intent


ROOT = Path(__file__).parent.parent
CASES_DIR = Path(__file__).parent
PIPELINE_CASES_PATH = CASES_DIR / "cases.json"
EMBED_CASES_PATH = CASES_DIR / "embed_cases.json"

_PIPELINES: dict[str, Pipeline] = {}
_COLD_STARTS: dict[str, float] = {}
_LATENCIES: dict[str, list[float]] = {}


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_id(case: dict[str, Any]) -> str:
    return case["id"]


def _dedupe_by_text(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for case in cases:
        if case["text"] in seen:
            continue
        seen.add(case["text"])
        unique.append(case)
    return unique


EMBED_CASES = load_cases(EMBED_CASES_PATH) if EMBED_CASES_PATH.exists() else []
PIPELINE_CASES = load_cases(PIPELINE_CASES_PATH)
ALL_CASES = _dedupe_by_text(EMBED_CASES + PIPELINE_CASES)
LLM_REPEATS = 3


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def run_embed_case(case: dict[str, Any]) -> None:
    resolver = get_resolver()
    command = resolver.resolve(Intent.from_dict(case["intent"]))
    expected_id = case["expected"]["device_id"]

    if command.device_id == expected_id:
        return

    pytest.fail(
        json.dumps(
            {
                "id": case["id"],
                "text": case["text"],
                "intent": case["intent"],
                "expected": expected_id,
                "actual": command.device_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        pytrace=False,
    )


def get_pipeline(model: str) -> Pipeline:
    pipeline = _PIPELINES.get(model)
    if pipeline is not None:
        return pipeline

    pipeline = default_pipeline(model=model, cache=False)
    _LATENCIES[model] = []

    started = time.perf_counter()
    pipeline.warmup()
    _COLD_STARTS[model] = time.perf_counter() - started

    _PIPELINES[model] = pipeline
    return pipeline


def run_case(model: str, case: dict[str, Any]) -> None:
    pipeline = get_pipeline(model)
    failures: list[dict[str, Any]] = []

    for attempt in range(1, LLM_REPEATS + 1):
        started = time.perf_counter()
        result = (
            pipeline
            .from_text(case["text"])
            .into(Expect(case["expected"]))
            .run()
        )
        _LATENCIES[model].append(time.perf_counter() - started)

        if result.ok:
            continue

        failures.append(
            {
                "attempt": attempt,
                "error": json.loads(result.error) if result.error else None,
            }
        )

    if not failures:
        return

    pytest.fail(
        json.dumps(
            {
                "id": case["id"],
                "input": case["text"],
                "repeats": LLM_REPEATS,
                "failed": len(failures),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        pytrace=False,
    )


def print_latency_stats() -> None:
    for model, values in _LATENCIES.items():
        if not values:
            continue

        print()
        print("=" * 56)
        print(f"Pipeline latency: {model}")
        print("=" * 56)
        print(f"cold start : {_COLD_STARTS[model]:.3f} s")
        print()
        print(f"requests   : {len(values)}")
        print(f"mean       : {statistics.fmean(values):.3f} s")
        print(f"min        : {min(values):.3f} s")
        print(f"p50        : {percentile(values, 50):.3f} s")
        print(f"p90        : {percentile(values, 90):.3f} s")
        print(f"p95        : {percentile(values, 95):.3f} s")
        print(f"p99        : {percentile(values, 99):.3f} s")
        print(f"max        : {max(values):.3f} s")
        print("=" * 56)


def close_pipelines() -> None:
    print_latency_stats()
    close_resources()
    _PIPELINES.clear()


atexit.register(close_pipelines)
