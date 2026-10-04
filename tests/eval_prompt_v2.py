"""One frozen run of semantic-prompt-v2. Prompt is already frozen. Does not change it."""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
from collections import Counter
from pathlib import Path

from ollama_runner.factory import semantic_pipeline
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.slots import read_slots, spoken_mention
from ollama_runner.sinks.expect import Expect
from ollama_runner.types import Command, Result


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "semantic_prompt_v2.json"
OLD = ROOT / "tests" / "intent_authoritative_v1.json"
HIGH = ("B-03", "B-12", "B-18", "B-20", "B-38")
PREVIOUS_WRONG = (
    "B-03", "B-12", "B-14", "B-18", "B-20", "B-38", "B-45", "B-48", "B-49",
    "B-55", "B-61", "B-62", "B-68", "B-79", "B-80", "B-81", "B-82", "B-92",
)
FALSE_REJECTS = ("B-16", "B-70", "B-71", "B-73", "B-76", "B-90", "B-91")
MEDIA = ("B-79", "B-80", "B-81", "B-82", "B-92")
FIELDS = ("device_type", "mention", "owner", "area", "ordinal")
OUTCOMES = ("command", "not_command", "needs_context")
FALSE_ACCEPT = (
    "Перескочи минуту вперёд",
    "Поставь послушать Элидар",
    "Запусти двигатель обсуждения",
)


class _Sink:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency(values: list[float]) -> dict:
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


def _rates(rows: list[dict], label: str) -> dict:
    expected = [row for row in rows if row["expected_outcome"] == label]
    predicted = [row for row in rows if row["predicted_outcome"] == label]
    tp = [row for row in expected if row["predicted_outcome"] == label]
    return {
        "expected": len(expected),
        "predicted": len(predicted),
        "tp": len(tp),
        "precision": None if not predicted else len(tp) / len(predicted),
        "recall": None if not expected else len(tp) / len(expected),
    }


def main() -> None:
    registry = DeviceRegistry.from_static()
    mapping = {
        row["id"]: row
        for row in json.loads((ROOT / "tests" / "reject_outcome_mapping.json").read_text(encoding="utf-8"))["cases"]
    }
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as response:
            resident = json.loads(response.read().decode()).get("models") or []
    except Exception as error:  # noqa: BLE001
        resident = [{"error": str(error)}]

    pipeline = semantic_pipeline()
    before = pipeline._nlu.calls
    started = time.perf_counter()
    pipeline.warmup()
    warmup_s = time.perf_counter() - started
    warmup_calls = pipeline._nlu.calls - before

    regression_rows = []
    for case in _load_regression():
        calls = pipeline._nlu.calls
        expected = case.get("semantic_expected", case["expected"])
        result = pipeline.run(case["text"], Expect(expected))
        obs = pipeline.observations[-1]
        resolution = case.get("resolution")
        ok = _resolution_matches(result, resolution) if resolution else result.ok
        regression_rows.append({
            "id": case["id"],
            "ok": ok,
            "fallback": not obs["handled"],
            "llm_calls": pipeline._nlu.calls - calls,
        })

    challenge = json.loads((ROOT / "tests" / "challenge_cases.json").read_text(encoding="utf-8"))
    challenge_rows = []
    for index, case in enumerate(challenge["cases"], start=1):
        calls = pipeline._nlu.calls
        result = pipeline.run(case["text"], _Sink())
        obs = pipeline.observations[-1]
        expected = mapping[case["id"]]
        if obs["handled"]:
            predicted = "command"
        else:
            predicted = obs["outcome"]
        final_intent = obs["final"]["intent"] if obs["final"] else None
        raw_intent = obs["llm"]["intent"] if obs["llm"] else None
        evidence = _text_slots(case["text"], registry)
        intent_correct = predicted == "command" and final_intent == expected["intent"]
        outcome_correct = predicted == expected["outcome"] and (
            expected["outcome"] != "command" or intent_correct
        )
        row = {
            "id": case["id"],
            "category": case["category"],
            "text": case["text"],
            "expected_outcome": expected["outcome"],
            "expected_intent": expected["intent"],
            "predicted_outcome": predicted,
            "deterministic_handled": obs["handled"],
            "fallback_called": not obs["handled"],
            "raw_intent": raw_intent,
            "final_intent": final_intent,
            "merge_changed_intent": obs["merge_changed_intent"],
            "merge_notes": obs["merge_notes"],
            "discarded_intent": obs["discarded_intent"],
            "llm": obs["llm"],
            "final": obs["final"],
            "blocked": _blocked(obs["llm"], obs["final"], evidence) if predicted == "command" else [],
            "resolver_status": obs["resolver_status"],
            "executed": predicted == "command",
            "outcome_correct": predicted == expected["outcome"],
            "intent_correct": intent_correct,
            "pipeline_correct": outcome_correct,
            "latency_s": obs["total_s"],
            "fallback_s": obs["fallback_s"],
            "llm_calls": pipeline._nlu.calls - calls,
        }
        challenge_rows.append(row)
        if index % 20 == 0 or row["fallback_called"]:
            print(
                f"[{index:03d}] {case['id']} {predicted} "
                f"{'ok' if row['pipeline_correct'] else 'miss'} {obs['fallback_s']:.2f}s",
                flush=True,
            )

    compound = json.loads((ROOT / "tests" / "compound_cases.json").read_text(encoding="utf-8"))
    compound_rows = []
    for case in compound["cases"]:
        calls = pipeline._nlu.calls
        pipeline.run(case["text"], _Sink())
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
        _, structural_extra, structural_missing = _match_commands(case["commands"], structural_actual)
        structural_correct = (
            structural_status == case["expected_status"]
            and structural_extra == 0
            and structural_missing == 0
        )
        if obs["handled"]:
            executed = structural_actual
            predicted = "command"
        elif obs["outcome"] == "command" and obs["final"]:
            executed = [{
                "intent": obs["final"]["intent"],
                "device_type": obs["final"]["device_type"],
                "area": obs["final"]["area"],
                "owner": obs["final"]["owner"],
            }]
            predicted = "command"
        else:
            executed = []
            predicted = obs["outcome"]
        _, extra, missing = _match_commands(case["commands"], executed)
        compound_rows.append({
            "id": case["id"],
            "text": case["text"],
            "expected_status": case["expected_status"],
            "structural_status": structural_status,
            "structural_correct": structural_correct,
            "structural_extra": structural_extra,
            "structural_missing": structural_missing,
            "fallback_called": not obs["handled"],
            "predicted_outcome": predicted,
            "executed": executed,
            "executed_extra": extra,
            "executed_missing": missing,
            "llm_calls": pipeline._nlu.calls - calls,
            "fallback_s": obs["fallback_s"],
            "latency_s": obs["total_s"],
        })
        print(f"compound {case['id']} {predicted} struct={'ok' if structural_correct else 'miss'}", flush=True)

    summary = _summarize(regression_rows, challenge_rows, compound_rows, warmup_s, warmup_calls, pipeline._nlu.calls, resident)
    summary["compare"] = _compare(challenge_rows, summary)
    OUT.write_text(
        json.dumps(
            {
                "summary": summary,
                "regression": regression_rows,
                "challenge": challenge_rows,
                "compound": compound_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _summarize(regression, challenge, compound, warmup_s, warmup_calls, llm_calls, resident) -> dict:
    matrix = {exp: {pred: 0 for pred in OUTCOMES} for exp in OUTCOMES}
    for row in challenge:
        matrix[row["expected_outcome"]][row["predicted_outcome"]] += 1
    fb = [row for row in challenge if row["fallback_called"]]
    paraphrase = [row for row in challenge if row["category"] == "semantic_paraphrase"]
    paraphrase_fb = [row for row in paraphrase if row["fallback_called"]]
    robustness_fb = [
        row for row in challenge
        if row["category"] == "parser_robustness" and row["fallback_called"]
    ]
    unsafe = [
        row for row in challenge
        if row["expected_outcome"] == "not_command" and row["predicted_outcome"] == "command"
    ]
    false_reject = [
        row for row in challenge
        if row["expected_outcome"] == "command" and row["predicted_outcome"] != "command"
    ]
    blocked = [item for row in fb if row["predicted_outcome"] == "command" for item in row["blocked"]]
    invented_kept = []
    for row in fb:
        if row["predicted_outcome"] != "command":
            continue
        final = row["final"] or {}
        raw = row["llm"] or {}
        evidence = _text_slots(row["text"], DeviceRegistry.from_static())
        for field in FIELDS:
            if _blank(evidence.get(field)) and not _blank(raw.get(field)) and not _blank(final.get(field)):
                invented_kept.append({"id": row["id"], "field": field, "llm": raw.get(field), "final": final.get(field)})

    def _brief(row: dict) -> dict:
        return {
            "id": row["id"],
            "text": row["text"],
            "predicted": row["predicted_outcome"],
            "intent": row["final_intent"],
            "deterministic": row["deterministic_handled"],
            "raw_intent": row["raw_intent"],
        }

    raw_correct = sum(
        row["predicted_outcome"] == "command" and row["raw_intent"] == row["expected_intent"]
        for row in paraphrase_fb
    )
    final_correct = sum(row["intent_correct"] for row in paraphrase_fb)
    to_incorrect = sum(
        row["predicted_outcome"] == "command"
        and row["raw_intent"] == row["expected_intent"]
        and row["final_intent"] != row["expected_intent"]
        for row in paraphrase_fb
    )
    to_correct = sum(
        row["predicted_outcome"] == "command"
        and row["raw_intent"] != row["expected_intent"]
        and row["final_intent"] == row["expected_intent"]
        for row in paraphrase_fb
    )
    by_outcome = {
        label: _latency([row["latency_s"] for row in fb if row["predicted_outcome"] == label])
        for label in OUTCOMES
    }
    return {
        "resident": [item.get("name") for item in resident if isinstance(item, dict)],
        "warmup_s": warmup_s,
        "warmup_calls": warmup_calls,
        "llm_calls": llm_calls,
        "regression": {
            "n": len(regression),
            "correct": sum(row["ok"] for row in regression),
            "fallback": sum(row["fallback"] for row in regression),
            "llm_calls": sum(row["llm_calls"] for row in regression),
        },
        "confusion": matrix,
        "rates": {label: _rates(challenge, label) for label in OUTCOMES},
        "pipeline_correct": sum(row["pipeline_correct"] for row in challenge),
        "outcome_correct": sum(row["outcome_correct"] for row in challenge),
        "fallback_requests": len(fb),
        "fallback_outcome_correct": sum(row["outcome_correct"] for row in fb),
        "fallback_pipeline_correct": sum(row["pipeline_correct"] for row in fb),
        "unsafe": [_brief(row) for row in unsafe],
        "false_reject": [_brief(row) for row in false_reject],
        "categories": {
            category: {
                "n": sum(row["category"] == category for row in challenge),
                "pipeline_correct": sum(row["pipeline_correct"] for row in challenge if row["category"] == category),
                "outcome_correct": sum(row["outcome_correct"] for row in challenge if row["category"] == category),
            }
            for category in ("parser_robustness", "semantic_paraphrase", "ambiguous", "adversarial")
        },
        "robustness_fallback": {
            "n": len(robustness_fb),
            "command": sum(row["predicted_outcome"] == "command" for row in robustness_fb),
            "false_rejected": sum(row["predicted_outcome"] != "command" for row in robustness_fb),
            "intent_correct": sum(row["intent_correct"] for row in robustness_fb),
        },
        "paraphrase": {
            "n": len(paraphrase),
            "pipeline_correct": sum(row["pipeline_correct"] for row in paraphrase),
            "fallback": len(paraphrase_fb),
            "command": sum(row["predicted_outcome"] == "command" for row in paraphrase_fb),
            "false_rejected": sum(row["predicted_outcome"] != "command" for row in paraphrase_fb),
            "correct_intent": sum(row["intent_correct"] for row in paraphrase_fb),
            "wrong_intent": sum(
                row["predicted_outcome"] == "command" and not row["intent_correct"] for row in paraphrase_fb
            ),
        },
        "ambiguous_confusion": _submatrix(challenge, "ambiguous"),
        "adversarial_command": [
            _brief(row) for row in challenge
            if row["category"] == "adversarial" and row["predicted_outcome"] == "command"
        ],
        "false_acceptance": [
            _brief(row) for row in challenge if row["text"] in FALSE_ACCEPT
        ],
        "merge": {
            "raw_correct": raw_correct,
            "final_correct": final_correct,
            "correct_to_incorrect": to_incorrect,
            "incorrect_to_correct": to_correct,
            "changes": [
                _brief(row) for row in paraphrase_fb if row["merge_changed_intent"]
            ],
        },
        "hallucinations": {
            "blocked_dropped": sum(item["kind"] == "dropped_absent" for item in blocked),
            "blocked_cases": sum(
                any(item["kind"] == "dropped_absent" for item in row["blocked"])
                for row in fb
                if row["predicted_outcome"] == "command"
            ),
            "replaced": sum(item["kind"] == "replaced_by_text" for item in blocked),
            "survived": invented_kept,
            "decline_with_discarded_intent": [
                {"id": row["id"], "outcome": row["predicted_outcome"], "intent": row["discarded_intent"]}
                for row in fb
                if row["predicted_outcome"] != "command" and row["discarded_intent"]
            ],
            "decline_with_final_target": [
                row["id"] for row in fb
                if row["predicted_outcome"] != "command" and row["final"] is not None
            ],
        },
        "compound": {
            "n": len(compound),
            "fully_deterministic": sum(not row["fallback_called"] for row in compound),
            "fallback": sum(row["fallback_called"] for row in compound),
            "structural_correct": sum(row["structural_correct"] for row in compound),
            "structural_extra": sum(row["structural_extra"] for row in compound),
            "structural_missing": sum(row["structural_missing"] for row in compound),
            "executed_extra": sum(row["executed_extra"] for row in compound),
            "executed_missing": sum(row["executed_missing"] for row in compound),
            "fallback_outcomes": dict(Counter(row["predicted_outcome"] for row in compound if row["fallback_called"])),
        },
        "latency": {
            **by_outcome,
            "overall": _latency([row["latency_s"] for row in challenge]),
            "fallback": _latency([row["latency_s"] for row in fb]),
            "deterministic": _latency([row["latency_s"] for row in challenge if row["deterministic_handled"]]),
        },
        "challenge_llm_calls": sum(row["llm_calls"] for row in challenge),
        "compound_llm_calls": sum(row["llm_calls"] for row in compound),
    }


def _view(row: dict) -> dict:
    return {
        "id": row["id"],
        "text": row["text"],
        "expected": row["expected_intent"],
        "outcome": row["predicted_outcome"],
        "intent": row["final_intent"],
        "resolver": row["resolver_status"],
        "deterministic": row["deterministic_handled"],
        "correct": row["pipeline_correct"],
    }


def _compare(rows: list[dict], summary: dict) -> dict:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    old_by_id = {row["id"]: row for row in old["challenge"]}
    new_by_id = {row["id"]: row for row in rows}

    def pair(ids: tuple[str, ...]) -> list[dict]:
        table = []
        for case_id in ids:
            previous = old_by_id[case_id]
            current = new_by_id[case_id]
            table.append({
                "id": case_id,
                "text": current["text"],
                "expected": current["expected_intent"],
                "old_outcome": previous["predicted_outcome"],
                "old_intent": previous["final_intent"],
                "old_resolver": previous["resolver_status"],
                "old_correct": previous["pipeline_correct"],
                "new_outcome": current["predicted_outcome"],
                "new_intent": current["final_intent"],
                "new_resolver": current["resolver_status"],
                "new_correct": current["pipeline_correct"],
            })
        return table

    eighteen = pair(PREVIOUS_WRONG)
    fixed = [row for row in eighteen if row["new_correct"]]
    false_rejected = [
        row for row in eighteen
        if not row["new_correct"] and row["new_outcome"] != "command"
    ]
    other_wrong = [
        row for row in eighteen
        if not row["new_correct"] and row["new_outcome"] == "command" and row["new_intent"] != row["old_intent"]
    ]
    unchanged = [
        row for row in eighteen
        if not row["new_correct"] and row["new_outcome"] == "command" and row["new_intent"] == row["old_intent"]
    ]
    regressions = []
    for case_id, previous in old_by_id.items():
        current = new_by_id[case_id]
        if previous["pipeline_correct"] and not current["pipeline_correct"]:
            regressions.append({
                **_view(current),
                "old_outcome": previous["predicted_outcome"],
                "old_intent": previous["final_intent"],
                "old_resolver": previous["resolver_status"],
            })
    new_unsafe_regressions = [
        row for row in regressions
        if row["outcome"] == "command" and row["resolver"] == "resolved"
    ]
    return {
        "high": pair(HIGH),
        "eighteen": {
            "fixed": [row["id"] for row in fixed],
            "unchanged": [row["id"] for row in unchanged],
            "became_false_reject": [row["id"] for row in false_rejected],
            "other_wrong_intent": [row["id"] for row in other_wrong],
            "rows": eighteen,
        },
        "false_rejects": pair(FALSE_REJECTS),
        "media": pair(MEDIA),
        "regressions": regressions,
        "new_unsafe_executable_regressions": new_unsafe_regressions,
        "old_confusion": old["summary"]["confusion"],
        "old_pipeline_correct": old["summary"]["pipeline_correct"],
        "old_paraphrase": old["summary"]["paraphrase"],
        "old_unsafe": old["summary"]["unsafe"],
        "old_categories": old["summary"]["categories"],
        "old_compound": old["summary"]["compound"],
        "old_hallucinations_survived": old["summary"]["hallucinations"]["survived"],
        "old_latency": old["summary"]["latency"],
        "intent_changed_by_merge": [
            row["id"] for row in rows
            if row["fallback_called"] and row["predicted_outcome"] == "command" and row["merge_changed_intent"]
        ],
    }


def _submatrix(rows: list[dict], category: str) -> dict:
    matrix = {exp: Counter() for exp in OUTCOMES}
    for row in rows:
        if row["category"] != category:
            continue
        matrix[row["expected_outcome"]][row["predicted_outcome"]] += 1
    return {exp: dict(matrix[exp]) for exp in OUTCOMES}


if __name__ == "__main__":
    main()
