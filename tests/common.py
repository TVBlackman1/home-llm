from __future__ import annotations

import atexit
import dataclasses
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ollama_runner.factory import (
    close_resources,
    default_pipeline,
    get_resolver,
    semantic_pipeline,
)
from ollama_runner.pipeline import Pipeline
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.sinks.expect import Expect
from ollama_runner.types import Command, Intent, Result


ROOT = Path(__file__).parent.parent
CASES_DIR = Path(__file__).parent
PIPELINE_CASES_PATH = CASES_DIR / "cases.json"
EMBED_CASES_PATH = CASES_DIR / "embed_cases.json"

_PIPELINES: dict[str, Pipeline] = {}
_SEMANTIC_PIPELINES: dict[str, SemanticPipeline] = {}
_COLD_STARTS: dict[str, float] = {}
_LATENCIES: dict[str, list[float]] = {}
_CASE_OUTCOMES: dict[str, Counter[str]] = {}


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


def get_semantic_pipeline(model: str) -> SemanticPipeline:
    key = f"{model} semantic"
    pipeline = _SEMANTIC_PIPELINES.get(key)
    if pipeline is not None:
        return pipeline

    pipeline = semantic_pipeline(model=model)
    _LATENCIES[key] = []

    started = time.perf_counter()
    pipeline.warmup()
    _COLD_STARTS[key] = time.perf_counter() - started

    _SEMANTIC_PIPELINES[key] = pipeline
    return pipeline


def _failure_detail(result: Result) -> Any:
    payload = result.payload or {}
    if not result.error:
        return None
    try:
        parsed = json.loads(result.error)
    except json.JSONDecodeError:
        parsed = {
            "error": result.error,
            "command": result.command.as_dict(),
        }
    if isinstance(parsed, dict):
        parsed.setdefault("status", payload.get("status"))
        parsed.setdefault("reason", payload.get("reason"))
        parsed.setdefault("candidates", payload.get("candidates"))
        parsed.setdefault("semantic", payload.get("semantic"))
    return parsed


def _resolution_matches(result: Result, resolution: dict[str, Any] | None) -> bool:
    if not resolution:
        return False
    payload = result.payload or {}
    if payload.get("status") != resolution.get("status"):
        return False
    if "reason" in resolution and (payload.get("reason") or "") != resolution["reason"]:
        return False
    semantic = payload.get("semantic") or {}
    for key in ("content", "owner", "area"):
        if key not in resolution:
            continue
        if (semantic.get(key) or "") != resolution[key]:
            return False
    return True


def _outcome_label(result: Result) -> str:
    payload = result.payload or {}
    status = str(payload.get("status") or "missing")
    reason = payload.get("reason") or ""
    if reason and status in {"clarify", "ambiguous", "not_found", "unsupported"}:
        return f"{status}/{reason}"
    return status


class _Capture:
    def __init__(self) -> None:
        self.commands: list[Command] = []

    def send(self, command: Command) -> Result:
        self.commands.append(command)
        return Result(ok=True, command=command)


def _targets_match(commands: list[Command], expected: dict[str, Any], targets: list[Any]) -> bool:
    """A scope fans out to a set of devices. The singular device_id is not the contract."""

    if {command.device_id for command in commands} != set(targets):
        return False
    if len(commands) != len(set(targets)):
        return False
    for command in commands:
        for key in ("action", "value", "owner", "place"):
            if key in expected and getattr(command, key) != expected[key]:
                return False
    return True


def run_semantic_case(model: str, case: dict[str, Any]) -> None:
    key = f"{model} semantic"
    pipeline = get_semantic_pipeline(model)
    failures: list[dict[str, Any]] = []
    last_label = "missing"

    for attempt in range(1, LLM_REPEATS + 1):
        started = time.perf_counter()
        expected = case.get("semantic_expected", case["expected"])
        targets = case.get("targets")
        if targets is not None:
            capture = _Capture()
            result = pipeline.run(case["text"], capture)
            _LATENCIES[key].append(time.perf_counter() - started)
            last_label = _outcome_label(result)
            resolution = case.get("resolution")
            matched = _targets_match(capture.commands, expected, targets)
            if targets:
                matched = matched and result.ok
            if resolution and not _resolution_matches(result, resolution):
                matched = False
            if matched:
                continue
            failures.append(
                {
                    "attempt": attempt,
                    "error": {
                        "targets": [command.device_id for command in capture.commands],
                        "expected_targets": targets,
                        "status": (result.payload or {}).get("status"),
                        "reason": (result.payload or {}).get("reason"),
                    },
                }
            )
            continue

        result = pipeline.run(case["text"], Expect(expected))
        _LATENCIES[key].append(time.perf_counter() - started)
        last_label = _outcome_label(result)

        resolution = case.get("resolution")
        if resolution:
            if _resolution_matches(result, resolution):
                continue
            detail = _failure_detail(result) or {}
            if isinstance(detail, dict):
                detail = {
                    **detail,
                    "expected_resolution": resolution,
                }
            failures.append(
                {
                    "attempt": attempt,
                    "error": detail,
                }
            )
            continue

        if result.ok:
            continue

        failures.append(
            {
                "attempt": attempt,
                "error": _failure_detail(result),
            }
        )

    if not failures:
        _CASE_OUTCOMES.setdefault(key, Counter())[last_label] += 1
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
        outcomes = _CASE_OUTCOMES.get(model)
        if outcomes:
            print()
            print(f"passing cases : {sum(outcomes.values())}")
            for label, count in sorted(outcomes.items(), key=lambda item: (-item[1], item[0])):
                print(f"  {label:<28} {count}")
        print("=" * 56)


def _print_nlu_split(pipeline) -> None:
    traces = getattr(pipeline, "traces", None) or []
    if not traces:
        return
    handled = [item for item in traces if item.handled]
    fallback = [item for item in traces if not item.handled]
    calls = getattr(pipeline._nlu, "calls", None)
    print()
    print("deterministic handled :", len(handled))
    print("fallback requests     :", len(fallback))
    if traces:
        print(f"fallback rate         : {100 * len(fallback) / len(traces):.1f}%")
    if calls is not None:
        print(f"llm calls             : {calls}")
        print(f"calls/request         : {calls / len(traces):.3f}")
    reasons: Counter[str] = Counter(item.reason for item in fallback)
    if reasons:
        print("fallback reasons:")
        for label, count in reasons.most_common():
            print(f"  {label:<24} {count}")
    intents: Counter[str] = Counter(item.final_intent for item in fallback)
    if intents:
        print("fallback intents:")
        for label, count in intents.most_common():
            print(f"  {label:<24} {count}")

    def _show(title: str, values: list[float]) -> None:
        if not values:
            return
        print(f"{title}")
        print(f"  n    : {len(values)}")
        print(f"  mean : {statistics.fmean(values):.3f} s")
        print(f"  p50  : {percentile(values, 50):.3f} s")
        print(f"  p90  : {percentile(values, 90):.3f} s")
        print(f"  p95  : {percentile(values, 95):.3f} s")
        print(f"  p99  : {percentile(values, 99):.3f} s")

    _show("deterministic latency", [item.deterministic_s for item in traces])
    _show("fallback latency", [item.fallback_s for item in fallback])
    _show("overall latency", [item.total_s for item in traces])
    if len(traces) >= 100:
        Path("ministral3.semantic.det-first.traces.json").write_text(
            json.dumps([dataclasses.asdict(item) for item in traces], ensure_ascii=False),
            encoding="utf-8",
        )


def close_pipelines() -> None:
    for pipeline in _SEMANTIC_PIPELINES.values():
        calls = getattr(pipeline._nlu, "calls", None)
        if calls is not None:
            print(f"llm calls  : {calls}")
        _print_nlu_split(pipeline)
    print_latency_stats()
    close_resources()
    _PIPELINES.clear()
    _SEMANTIC_PIPELINES.clear()


atexit.register(close_pipelines)
