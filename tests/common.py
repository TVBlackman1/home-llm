from __future__ import annotations

import atexit
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from ollama_runner.main import OllamaRunner


ROOT = Path(__file__).parent.parent

PROMPT_PATH = ROOT / "src" / "ollama_runner" / "prompts" / "homeassist.md"
CASES_PATH = Path(__file__).parent / "cases.json"

# Один runner на модель в рамках pytest-процесса.
_RUNNERS: dict[str, OllamaRunner] = {}

# Первый запрос: загрузка модели + первый inference.
_COLD_STARTS: dict[str, float] = {}

# Latency реальных тестовых запросов.
_LATENCIES: dict[str, list[float]] = {}


def load_cases() -> list[dict[str, Any]]:
    return json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )


CASES = load_cases()


def case_id(case: dict[str, Any]) -> str:
    return case["text"]


def percentile(values: list[float], p: float) -> float:
    """Percentile с линейной интерполяцией без numpy."""
    if not values:
        raise ValueError("percentile() requires at least one value")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p / 100.0

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def get_runner(model: str) -> OllamaRunner:
    runner = _RUNNERS.get(model)

    if runner is not None:
        return runner

    runner = OllamaRunner(
        model=model,
        prompt_path=PROMPT_PATH,
        temperature=0.0,
        keep_alive="10m",
    )

    runner.__enter__()

    _RUNNERS[model] = runner
    _LATENCIES[model] = []

    # Первый запрос:
    # - загружает модель в RAM/VRAM;
    # - прогревает inference;
    # - не входит в latency тесткейсов.
    started = time.perf_counter()

    runner.run("Включи свет")

    _COLD_STARTS[model] = time.perf_counter() - started

    return runner


def run_case(
    model: str,
    case: dict[str, Any],
) -> None:
    runner = get_runner(model)

    started = time.perf_counter()

    result = runner.run(case["text"])

    elapsed = time.perf_counter() - started

    _LATENCIES[model].append(elapsed)

    expected = case["expected"]

    if result != expected:
        diff = {
            key: {
                "expected": expected.get(key),
                "actual": result.get(key),
            }
            for key in expected.keys() | result.keys()
            if expected.get(key) != result.get(key)
        }

        pytest.fail(
            json.dumps(
                {
                    "input": case["text"],
                    "diff": diff,
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
        print(f"Ollama latency: {model}")
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


def close_runners() -> None:
    print_latency_stats()

    for runner in _RUNNERS.values():
        runner.__exit__(None, None, None)


atexit.register(close_runners)