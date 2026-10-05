"""Frozen baseline: production SemanticPipeline on 279, 220, and 65.

Does not change grammar, prompt, schema, registry, resolver, or expected labels.
Records the raw Ministral command and the command after deterministic merge.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from ollama_runner.factory import semantic_pipeline
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.slots import read_slots, spoken_mention
from ollama_runner.sinks.expect import Expect
from ollama_runner.types import Command, Result


class _Capture:
    def __init__(self) -> None:
        self.commands: list[Command] = []

    def send(self, command: Command) -> Result:
        self.commands.append(command)
        return Result(ok=True, command=command)


def _targets_match(commands: list[Command], expected: dict, targets: list) -> bool:
    if {command.device_id for command in commands} != set(targets):
        return False
    if len(commands) != len(set(targets)):
        return False
    for command in commands:
        for key in ("action", "value", "owner", "place"):
            if key in expected and getattr(command, key) != expected[key]:
                return False
    return True


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fallback_baseline.json"
REJECT = "NEEDS_CONTEXT"
FIELDS = ("device_type", "mention", "owner", "area", "ordinal")


class _Sink:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100 * part / whole:.1f}%"


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _fmt(stats: dict) -> str:
    if not stats.get("n"):
        return "n=0"
    return (
        f"n={stats['n']} mean={stats['mean']:.3f} "
        f"p50={stats['p50']:.3f} p90={stats['p90']:.3f} "
        f"p95={stats['p95']:.3f} p99={stats['p99']:.3f}"
    )


def _blank(value) -> bool:
    return value is None or value == "" or value == 0


def _resolution_matches(result: Result, resolution: dict) -> bool:
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


def _load_regression() -> list[dict]:
    cases = json.loads((ROOT / "tests" / "embed_cases.json").read_text(encoding="utf-8"))
    cases += json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    unique = []
    for case in cases:
        if case["text"] in seen:
            continue
        seen.add(case["text"])
        unique.append(case)
    return unique


def _model_resident() -> dict:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as response:
            payload = json.loads(response.read().decode())
    except Exception as error:  # noqa: BLE001
        return {"error": str(error), "models": []}
    return {"models": payload.get("models") or []}


def _text_slots(text: str, registry: DeviceRegistry) -> dict:
    slots = read_slots(text, registry)
    return {
        "device_type": slots.device_type,
        "mention": spoken_mention(text, registry),
        "owner": slots.owner,
        "area": slots.area,
        "ordinal": slots.ordinal,
    }


def _blocked(llm: dict | None, final: dict | None, evidence: dict) -> list[dict]:
    if not llm or not final:
        return []
    blocked = []
    for field in FIELDS:
        raw = llm.get(field)
        if _blank(raw):
            continue
        kept = final.get(field)
        attested = evidence.get(field)
        if _blank(attested) and _blank(kept):
            blocked.append({"field": field, "llm": raw, "final": kept, "kind": "dropped_absent"})
        elif not _blank(attested) and kept != raw:
            blocked.append({"field": field, "llm": raw, "final": kept, "kind": "replaced_by_text"})
    return blocked


def _same_command(expected: dict, actual: dict) -> bool:
    for key in ("intent", "device_type", "area", "owner"):
        if key in expected and (expected.get(key) or None) != (actual.get(key) or None):
            return False
    return True


def _match_commands(expected: list[dict], actual: list[dict]) -> tuple[int, int, int]:
    used = [False] * len(actual)
    matched = 0
    for want in expected:
        for index, got in enumerate(actual):
            if used[index] or not _same_command(want, got):
                continue
            used[index] = True
            matched += 1
            break
    return matched, len(actual) - matched, len(expected) - matched


def main() -> None:
    registry = DeviceRegistry.from_static()
    resident_before = _model_resident()
    pipeline = semantic_pipeline()
    calls_before_warmup = pipeline._nlu.calls
    warmup_started = time.perf_counter()
    pipeline.warmup()
    warmup_s = time.perf_counter() - warmup_started
    warmup_calls = pipeline._nlu.calls - calls_before_warmup

    regression_rows = []
    for case in _load_regression():
        before = pipeline._nlu.calls
        expected = case.get("semantic_expected", case["expected"])
        targets = case.get("targets")
        if targets is not None:
            capture = _Capture()
            result = pipeline.run(case["text"], capture)
            ok = _targets_match(capture.commands, expected, targets)
            if targets:
                ok = ok and result.ok
            resolution = case.get("resolution")
            if resolution and not _resolution_matches(result, resolution):
                ok = False
        else:
            result = pipeline.run(case["text"], Expect(expected))
            resolution = case.get("resolution")
            ok = _resolution_matches(result, resolution) if resolution else result.ok
        obs = pipeline.observations[-1]
        regression_rows.append({
            "id": case["id"],
            "text": case["text"],
            "ok": ok,
            "fallback": not obs["handled"],
            "llm_calls": pipeline._nlu.calls - before,
            "final_intent": obs["final"]["intent"] if obs["final"] else None,
            "resolver": obs["resolver_status"],
        })

    challenge = json.loads((ROOT / "tests" / "challenge_cases.json").read_text(encoding="utf-8"))
    challenge_rows = []
    for index, case in enumerate(challenge["cases"], start=1):
        before = pipeline._nlu.calls
        started = time.perf_counter()
        result = pipeline.run(case["text"], _Sink())
        wall = time.perf_counter() - started
        obs = pipeline.observations[-1]
        evidence = _text_slots(case["text"], registry)
        expected = case["expected_intent"]
        det_intent = obs["plan_commands"][0]["intent"] if obs["handled"] and obs["plan_commands"] else None
        final_intent = obs["final"]["intent"] if obs["final"] else None
        if expected == REJECT:
            det_correct = not obs["handled"]
            pipe_correct = False
        else:
            det_correct = bool(obs["handled"] and det_intent == expected)
            pipe_correct = final_intent == expected
        fallback_correct = (not obs["handled"]) and pipe_correct
        row = {
            "id": case["id"],
            "category": case["category"],
            "text": case["text"],
            "expected": expected,
            "note": case.get("note", ""),
            "deterministic_handled": obs["handled"],
            "deterministic_result": obs["plan_commands"],
            "deterministic_reason": obs["reason"],
            "unresolved_spans": obs["unresolved_spans"],
            "fallback_called": not obs["handled"],
            "llm_raw": obs["llm"],
            "final": obs["final"],
            "merge_notes": obs["merge_notes"],
            "blocked": _blocked(obs["llm"], obs["final"], evidence),
            "text_slots": evidence,
            "resolver": obs["resolver"],
            "resolver_status": obs["resolver_status"],
            "resolver_reason": obs["resolver_reason"],
            "payload_status": (result.payload or {}).get("status"),
            "deterministic_correct": det_correct,
            "fallback_correct": fallback_correct,
            "pipeline_correct": pipe_correct,
            "false_acceptance": bool(obs["handled"] and not det_correct),
            "latency_s": obs["total_s"],
            "wall_s": wall,
            "deterministic_s": obs["deterministic_s"],
            "fallback_s": obs["fallback_s"],
            "llm_calls": pipeline._nlu.calls - before,
        }
        challenge_rows.append(row)
        if index % 20 == 0 or row["fallback_called"]:
            mark = "ok" if row["pipeline_correct"] else "miss"
            print(
                f"[{index:03d}/{len(challenge['cases'])}] {case['id']} "
                f"{'fb' if row['fallback_called'] else 'det'} {mark} "
                f"{final_intent} {obs['fallback_s']:.2f}s",
                flush=True,
            )

    compound = json.loads((ROOT / "tests" / "compound_cases.json").read_text(encoding="utf-8"))
    compound_rows = []
    for case in compound["cases"]:
        before = pipeline._nlu.calls
        result = pipeline.run(case["text"], _Sink())
        obs = pipeline.observations[-1]
        if obs["fully_parsed"] and obs["plan_commands"]:
            structural_status = "fully_parsed"
        elif obs["plan_commands"] and obs["unresolved_spans"]:
            structural_status = "partially_parsed"
        else:
            structural_status = "unhandled"
        structural_actual = [
            {
                "intent": command["intent"],
                "device_type": command["device_type"],
                "area": command["area"],
                "owner": command["owner"],
            }
            for command in obs["plan_commands"]
        ]
        structural_matched, structural_extra, structural_missing = _match_commands(
            case["commands"], structural_actual
        )
        structural_correct = (
            structural_status == case["expected_status"]
            and structural_extra == 0
            and structural_missing == 0
        )
        if obs["handled"]:
            executed = structural_actual
        elif obs["final"]:
            executed = [{
                "intent": obs["final"]["intent"],
                "device_type": obs["final"]["device_type"],
                "area": obs["final"]["area"],
                "owner": obs["final"]["owner"],
            }]
        else:
            executed = []
        matched, extra, missing = _match_commands(case["commands"], executed)
        plan_correct = extra == 0 and missing == 0 and (
            obs["handled"] or case["expected_status"] != "fully_parsed"
        )
        compound_rows.append({
            "id": case["id"],
            "category": case["category"],
            "text": case["text"],
            "expected_status": case["expected_status"],
            "expected": case["commands"],
            "structural_status": structural_status,
            "structural_actual": structural_actual,
            "structural_correct": structural_correct,
            "structural_extra": structural_extra,
            "structural_missing": structural_missing,
            "fully_deterministic": obs["handled"],
            "fallback_called": not obs["handled"],
            "executed": executed,
            "executed_extra": extra,
            "executed_missing": missing,
            "plan_correct": plan_correct and matched == len(case["commands"]),
            "policy_fallback": (not obs["handled"]) if not obs["fully_parsed"] else False,
            "llm_raw": obs["llm"],
            "final": obs["final"],
            "resolver": obs["resolver"],
            "resolver_status": (result.payload or {}).get("status"),
            "llm_calls": pipeline._nlu.calls - before,
            "latency_s": obs["total_s"],
            "fallback_s": obs["fallback_s"],
        })
        print(
            f"compound {case['id']} {'det' if obs['handled'] else 'fb'} "
            f"struct={'ok' if structural_correct else 'miss'}",
            flush=True,
        )

    summary = _summarize(
        regression_rows,
        challenge_rows,
        compound_rows,
        warmup_s,
        warmup_calls,
        pipeline._nlu.calls,
        resident_before,
    )
    payload = {
        "resident_before": resident_before,
        "warmup_s": warmup_s,
        "warmup_calls": warmup_calls,
        "llm_calls_total": pipeline._nlu.calls,
        "summary": summary,
        "regression": regression_rows,
        "challenge": challenge_rows,
        "compound": compound_rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _summarize(regression, challenge, compound, warmup_s, warmup_calls, llm_calls, resident) -> dict:
    det = [row for row in challenge if row["deterministic_handled"]]
    fb = [row for row in challenge if row["fallback_called"]]
    by_cat = {}
    for category in ("parser_robustness", "semantic_paraphrase", "ambiguous", "adversarial"):
        rows = [row for row in challenge if row["category"] == category]
        accepted = [row for row in rows if row["deterministic_handled"]]
        sent = [row for row in rows if row["fallback_called"]]
        by_cat[category] = {
            "n": len(rows),
            "deterministic_accepted": len(accepted),
            "deterministic_correct": sum(row["deterministic_correct"] for row in accepted),
            "deterministic_incorrect": sum(not row["deterministic_correct"] for row in accepted),
            "deterministic_rejected": len(sent),
            "fallback_requests": len(sent),
            "fallback_correct": sum(row["fallback_correct"] for row in sent),
            "fallback_incorrect": sum(not row["fallback_correct"] for row in sent),
            "pipeline_correct": sum(row["pipeline_correct"] for row in rows),
            "pipeline_incorrect": sum(not row["pipeline_correct"] for row in rows),
        }

    paraphrase = [row for row in fb if row["category"] == "semantic_paraphrase"]
    per_intent = {}
    for intent in sorted({row["expected"] for row in paraphrase}):
        rows = [row for row in paraphrase if row["expected"] == intent]
        per_intent[intent] = {
            "total": len(rows),
            "correct": sum(row["fallback_correct"] for row in rows),
            "wrong": sum(not row["fallback_correct"] for row in rows),
        }

    false_accept = [row for row in challenge if row["false_acceptance"]]
    wrong_intent = [
        row for row in fb
        if row["expected"] != REJECT and not row["fallback_correct"]
    ]
    forced_noncommand = [
        row for row in fb
        if row["category"] == "adversarial" and row["expected"] == REJECT
    ]
    forced_context = [
        row for row in fb
        if row["category"] == "ambiguous" and row["expected"] == REJECT
    ]
    classified = {id(row) for row in wrong_intent + forced_noncommand + forced_context}
    other = [row for row in fb if not row["pipeline_correct"] and id(row) not in classified]

    blocked = [item for row in fb for item in row["blocked"]]
    invented_kept = []
    for row in fb:
        evidence = row["text_slots"]
        final = row["final"] or {}
        raw = row["llm_raw"] or {}
        for field in FIELDS:
            if _blank(evidence.get(field)) and not _blank(raw.get(field)) and not _blank(final.get(field)):
                invented_kept.append({
                    "id": row["id"],
                    "text": row["text"],
                    "field": field,
                    "llm": raw.get(field),
                    "final": final.get(field),
                })

    robustness = [row for row in fb if row["category"] == "parser_robustness"]
    slot_ok = 0
    for row in robustness:
        final = row["final"] or {}
        evidence = row["text_slots"]
        if all((final.get(field) or None) == (evidence.get(field) or None) for field in FIELDS):
            slot_ok += 1

    det_latency = _latency([row["latency_s"] for row in det])
    fb_latency = _latency([row["latency_s"] for row in fb])
    fb_model = _latency([row["fallback_s"] for row in fb])
    all_latency = _latency([row["latency_s"] for row in challenge])

    structural_correct = sum(row["structural_correct"] for row in compound)
    return {
        "resident_before": [item.get("name") for item in resident.get("models", [])],
        "warmup_s": warmup_s,
        "warmup_calls": warmup_calls,
        "regression": {
            "n": len(regression),
            "correct": sum(row["ok"] for row in regression),
            "fallback": sum(row["fallback"] for row in regression),
            "llm_calls": sum(row["llm_calls"] for row in regression),
        },
        "challenge": {
            "n": len(challenge),
            "deterministic_accepted": len(det),
            "deterministic_correct": sum(row["deterministic_correct"] for row in det),
            "deterministic_incorrect": sum(not row["deterministic_correct"] for row in det),
            "fallback_requests": len(fb),
            "fallback_correct": sum(row["fallback_correct"] for row in fb),
            "fallback_incorrect": sum(not row["fallback_correct"] for row in fb),
            "pipeline_correct": sum(row["pipeline_correct"] for row in challenge),
            "pipeline_incorrect": sum(not row["pipeline_correct"] for row in challenge),
            "false_acceptance": [
                {"id": row["id"], "text": row["text"], "expected": row["expected"], "got": row["final"]["intent"]}
                for row in false_accept
            ],
        },
        "by_category": by_cat,
        "paraphrase_fallback": {
            "total": len(paraphrase),
            "correct": sum(row["fallback_correct"] for row in paraphrase),
            "wrong": sum(not row["fallback_correct"] for row in paraphrase),
            "per_intent": per_intent,
        },
        "taxonomy": {
            "deterministic_false_acceptance": len(false_accept),
            "wrong_llm_intent": len(wrong_intent),
            "non_command_forced": len(forced_noncommand),
            "needs_context_forced": len(forced_context),
            "other": len(other),
            "blocked_fields": len(blocked),
            "blocked_dropped_absent": sum(item["kind"] == "dropped_absent" for item in blocked),
            "blocked_replaced_by_text": sum(item["kind"] == "replaced_by_text" for item in blocked),
            "invented_kept": invented_kept,
        },
        "robustness_fallback": {
            "total": len(robustness),
            "intent_correct": sum(row["fallback_correct"] for row in robustness),
            "slots_match_text": slot_ok,
        },
        "compound": {
            "n": len(compound),
            "fully_deterministic": sum(row["fully_deterministic"] for row in compound),
            "fallback_required": sum(row["fallback_called"] for row in compound),
            "structural_correct": structural_correct,
            "structural_incorrect": len(compound) - structural_correct,
            "executed_plan_correct": sum(row["plan_correct"] for row in compound),
            "executed_extra": sum(row["executed_extra"] for row in compound),
            "executed_missing": sum(row["executed_missing"] for row in compound),
            "structural_extra": sum(row["structural_extra"] for row in compound),
            "structural_missing": sum(row["structural_missing"] for row in compound),
        },
        "latency": {
            "deterministic_requests": det_latency,
            "fallback_requests": fb_latency,
            "fallback_model_only": fb_model,
            "overall_challenge": all_latency,
        },
        "llm_calls_total_including_warmup": llm_calls,
        "challenge_llm_calls": sum(row["llm_calls"] for row in challenge),
        "compound_llm_calls": sum(row["llm_calls"] for row in compound),
    }


if __name__ == "__main__":
    main()
