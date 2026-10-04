"""Offline V2A.2 discourse-only experiment. Does not modify production or V2A."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.semantic import CommandOutcome, RequestContext
from ollama_runner.semantic_pipeline import _merge


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reject_v2",
    ROOT / "tests" / "eval_reject_semantics_v2.py",
)
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)

PROMPT = ROOT / "tests/prompts/reject_semantics_v2a2.md"
VARIANT = "v2a2"
SUMMARY = ROOT / "tests/reject_semantics_v2a2_summary.json"
REPEATS = 3

FOCUSED = [
    ("D-01", "Свет в конце тоннеля меня не включает", "safety"),
    ("D-03", "Телевизор в новостях опять показывает политику", "safety"),
    ("D-11", "Не включай телевизор", "safety"),
    ("D-25", "Сделай вид, что свет не горит", "safety"),
    ("obs-on", "Свет горит", "observation"),
    ("obs-off", "Свет не горит", "observation"),
    ("obs-tv", "Телевизор показывает политику", "observation"),
    ("A-23", "У Антона в кабинете монитор выключи", "polarity"),
    ("A-57", "У Маши, кстати, телевизор выключи", "polarity"),
    ("A-06", "Свет в спальне, будь добр, включи", "discourse"),
    ("syn-please", "Телевизор, пожалуйста, выключи", "discourse"),
    ("syn-bytheway", "Телевизор, кстати, выключи", "discourse"),
    ("syn-if", "Свет, если можно, включи", "discourse"),
    ("neg-on", "Кстати, свет горит", "neg-discourse"),
    ("neg-off", "Кстати, телевизор выключен", "neg-discourse"),
    ("neg-music", "Вообще-то музыка играет", "neg-discourse"),
    ("neg-news", "Между прочим, телевизор показывает новости", "neg-discourse"),
    ("pro-mid", "Телевизор не включай", "prohibition"),
    ("pro-please", "Пожалуйста, не включай телевизор", "prohibition"),
    ("pro-bytheway", "Кстати, телевизор не включай", "prohibition"),
    ("pret-bytheway", "Кстати, представь, что телевизор выключен", "pretend"),
    ("pret-please", "Пожалуйста, представь, что свет включён", "pretend"),
    ("B-25", "Свет режет глаза на кухне", "measure"),
    ("B-56", "Повремени с фильмом", "measure"),
    ("B-72", "Отмотай к прошлому треку", "measure"),
    ("B-88", "Давай Кварис", "measure"),
]

WATCH = ("A-06", "A-23", "A-57", "B-13", "B-22", "B-25", "B-56", "B-72", "B-88", "D-01", "D-03", "D-11", "D-25")


def _label(row: dict, expected: str) -> str:
    llm = row["llm"]
    outcome = llm.get("outcome")
    intent = row.get("final_intent") or llm.get("intent") or ""
    if expected == v2.REJECT:
        if row.get("handled"):
            return "det_false"
        if outcome == "command":
            return "exec" if row.get("resolver_status") == "resolved" else "false_cmd"
        return "reject_ok"
    if intent == expected and (row.get("handled") or outcome == "command"):
        return "correct"
    if outcome in {"not_command", "needs_context"} or intent == "":
        return "false_reject"
    return "wrong"


def _rows() -> list[dict]:
    return [json.loads(line) for line in v2.RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _require_saved(rows: list[dict]) -> None:
    for variant in ("production", "v2a"):
        count = sum(1 for row in rows if row.get("variant") == variant and row.get("split") == "challenge")
        if count != 220 * REPEATS:
            raise SystemExit(f"saved {variant} challenge rows={count}, expected {220 * REPEATS}")


def _run_focused(pipeline, recording, registry, resolver) -> None:
    have = v2._done(VARIANT, "focused")
    if have >= len(FOCUSED) * REPEATS:
        print("v2a2 focused already complete", flush=True)
        return
    start = have // REPEATS
    for index, (case_id, text, role) in enumerate(FOCUSED):
        if index < start:
            continue
        votes = []
        for attempt in range(1, REPEATS + 1):
            if index == start and (index * REPEATS + attempt) <= have:
                continue
            recording.last = None
            outcome = recording.parse(text, RequestContext())
            view = v2._outcome_view(outcome)
            status = None
            reason = None
            if isinstance(outcome, CommandOutcome):
                semantic, slots, notes = _merge(
                    text,
                    parse_deterministic(text, registry),
                    outcome.command,
                    registry,
                )
                resolved = resolver.resolve(
                    semantic,
                    context=RequestContext(),
                    text=text,
                    slots=slots,
                    nlu_intent=outcome.command.intent,
                    normalization=notes,
                )
                status = resolved.status
                reason = resolved.reason
            vote = view["outcome"] if view["outcome"] != "command" else f"command:{view['intent']}"
            votes.append(vote if status is None else f"{vote}|{status}")
            v2._append({
                "variant": VARIANT,
                "split": "focused",
                "id": case_id,
                "role": role,
                "text": text,
                "attempt": attempt,
                "llm": view,
                "resolver_status": status,
                "resolver_reason": reason,
            })
        print(f"v2a2 {case_id} {Counter(votes)} | {text}", flush=True)


def _metrics(rows: list[dict], cases: list[dict], variant: str) -> dict:
    by_case = {case["id"]: case for case in cases}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("variant") == variant and row.get("split") == "challenge":
            grouped.setdefault(row["id"], []).append(row)
    correct = false_cmd = exec_cmd = false_reject = wrong = unstable = 0
    false_c, false_d = [], []
    remaining, rejects, wrongs, unstable_ids = [], [], [], []
    fallback = fallback_correct = 0
    labels_by_id = {}
    any_exec = []
    for case_id, runs in grouped.items():
        case = by_case[case_id]
        expected = case["expected_intent"]
        sigs = [v2._signature(row) for row in runs]
        if len(set(sigs)) > 1:
            unstable += 1
            unstable_ids.append({"id": case_id, "votes": sigs, "text": case["text"]})
        labels = [_label(row, expected) for row in runs]
        vote = v2._majority(labels)
        labels_by_id[case_id] = {"vote": vote, "sigs": sigs, "expected": expected, "text": case["text"], "category": case["category"]}
        if any(not row["handled"] for row in runs):
            fallback += 1
            if vote == "correct" or (expected == v2.REJECT and vote == "reject_ok"):
                fallback_correct += 1
        if vote in {"correct", "reject_ok"}:
            correct += 1
        elif vote == "false_reject":
            false_reject += 1
            rejects.append({"id": case_id, "text": case["text"], "expected": expected, "votes": sigs})
        elif vote == "wrong":
            wrong += 1
            wrongs.append({"id": case_id, "text": case["text"], "expected": expected, "votes": sigs})
        if expected == v2.REJECT and vote in {"false_cmd", "exec"}:
            false_cmd += 1
            (false_c if case["category"] == "ambiguous" else false_d).append(case_id)
            executable = vote == "exec"
            if executable:
                exec_cmd += 1
            remaining.append({
                "id": case_id,
                "category": case["category"],
                "text": case["text"],
                "votes": sigs,
                "executable": executable,
                "resolver": [row.get("resolver_status") for row in runs],
            })
        if any(row.get("resolver_status") == "resolved" and row["llm"].get("outcome") == "command" and expected == v2.REJECT for row in runs):
            any_exec.append(case_id)
    return {
        "correct": correct,
        "n": len(grouped),
        "fallback": fallback,
        "fallback_correct": fallback_correct,
        "false_command": false_cmd,
        "false_command_c": false_c,
        "false_command_d": false_d,
        "executable_false": exec_cmd,
        "any_executable_run": any_exec,
        "false_rejects": false_reject,
        "wrong_intents": wrong,
        "unstable": unstable,
        "remaining": remaining,
        "reject_rows": rejects,
        "wrong_rows": wrongs,
        "unstable_rows": unstable_ids,
        "labels": labels_by_id,
    }


def _delta(base: dict, new: dict) -> dict:
    recovered, lost, new_false = [], [], []
    for case_id, item in new["labels"].items():
        old = base["labels"][case_id]
        if old["vote"] == "false_reject" and item["vote"] == "correct":
            recovered.append({"id": case_id, "text": item["text"], "expected": item["expected"], "votes": item["sigs"]})
        if old["vote"] == "correct" and item["vote"] != "correct":
            lost.append({"id": case_id, "text": item["text"], "was": old["sigs"], "now": item["sigs"], "vote": item["vote"]})
        if item["expected"] == v2.REJECT and item["vote"] in {"false_cmd", "exec"} and old["vote"] not in {"false_cmd", "exec", "det_false"}:
            new_false.append({"id": case_id, "text": item["text"], "votes": item["sigs"], "resolver": next(row["resolver"] for row in new["remaining"] if row["id"] == case_id)})
    return {"recovered": recovered, "lost_vs_base_correct": lost, "new_false_command": new_false}


def _focused_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("variant") == VARIANT and row.get("split") == "focused":
            grouped.setdefault(row["id"], []).append(row)
    out = {}
    for case_id, _text, role in FOCUSED:
        runs = grouped.get(case_id, [])
        votes = []
        for row in runs:
            llm = row["llm"]
            vote = llm["outcome"] if llm["outcome"] != "command" else f"command:{llm['intent']}"
            if row.get("resolver_status"):
                vote = f"{vote}|{row['resolver_status']}"
            votes.append(vote)
        out[case_id] = {"role": role, "text": runs[0]["text"] if runs else "", "votes": dict(Counter(votes))}
    return out


def _polarity(rows: list[dict], cases: list[dict], variant: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("variant") == variant and row.get("split") == "challenge" and not row.get("handled"):
            grouped.setdefault(row["id"], []).append(row)
    flips = []
    for case in cases:
        expected = case["expected_intent"]
        if expected not in {"device.turn_on", "device.turn_off"}:
            continue
        runs = grouped.get(case["id"])
        if not runs:
            continue
        opposite = "device.turn_off" if expected == "device.turn_on" else "device.turn_on"
        bad = 0
        for row in runs:
            intent = row.get("final_intent") or row["llm"].get("intent") or ""
            if row["llm"].get("outcome") == "command" and intent == opposite:
                bad += 1
        if bad:
            flips.append({
                "id": case["id"],
                "text": case["text"],
                "expected": expected,
                "opposite_runs": bad,
                "runs": len(runs),
            })
    return flips


def main() -> None:
    rows = _rows()
    _require_saved(rows)
    registry = v2.DeviceRegistry.from_static()
    challenge = v2._load_json(ROOT / "tests/challenge_cases.json")["cases"]
    print(f"=== {VARIANT} {PROMPT} ===", flush=True)
    pipeline, recording = v2._pipeline(PROMPT, registry)
    pipeline.warmup()
    _run_focused(pipeline, recording, registry, pipeline._resolver)
    v2._run_challenge(VARIANT, pipeline, recording, challenge, registry)
    recording.inner.close()
    rows = _rows()
    production = _metrics(rows, challenge, "production")
    v2a = _metrics(rows, challenge, "v2a")
    current = _metrics(rows, challenge, VARIANT)
    payload = {
        "production": {key: value for key, value in production.items() if key != "labels"},
        "v2a": {key: value for key, value in v2a.items() if key != "labels"},
        "v2a2": {key: value for key, value in current.items() if key != "labels"},
        "delta_v2a_to_v2a2": _delta(v2a, current),
        "delta_production_to_v2a2": _delta(production, current),
        "watch": {
            case_id: {
                "production": production["labels"][case_id]["sigs"],
                "v2a": v2a["labels"][case_id]["sigs"],
                "v2a2": current["labels"][case_id]["sigs"],
                "v2a2_vote": current["labels"][case_id]["vote"],
            }
            for case_id in WATCH
        },
        "focused": _focused_summary(rows),
        "polarity": {
            "production": _polarity(rows, challenge, "production"),
            "v2a": _polarity(rows, challenge, "v2a"),
            "v2a2": _polarity(rows, challenge, VARIANT),
        },
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {
        name: {
            "correct": item["correct"],
            "fallback_correct": item["fallback_correct"],
            "false_command": item["false_command"],
            "C": item["false_command_c"],
            "D": item["false_command_d"],
            "executable": item["executable_false"],
            "any_exec": item["any_executable_run"],
            "false_rejects": item["false_rejects"],
            "wrong": item["wrong_intents"],
            "unstable": item["unstable"],
        }
        for name, item in (("production", production), ("v2a", v2a), ("v2a2", current))
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    print("RECOVERED", json.dumps(payload["delta_v2a_to_v2a2"]["recovered"], ensure_ascii=False), flush=True)
    print("LOST VS PROD", json.dumps(payload["delta_production_to_v2a2"]["lost_vs_base_correct"], ensure_ascii=False), flush=True)
    print("NEW FALSE", json.dumps(payload["delta_v2a_to_v2a2"]["new_false_command"], ensure_ascii=False), flush=True)
    new_wrong = []
    for case_id, item in current["labels"].items():
        old = v2a["labels"][case_id]
        if item["vote"] == "wrong" and old["vote"] != "wrong":
            new_wrong.append({"id": case_id, "text": item["text"], "was": old["sigs"], "now": item["sigs"]})
    print("NEW WRONG", json.dumps(new_wrong, ensure_ascii=False), flush=True)
    print("POLARITY", json.dumps(payload["polarity"]["v2a2"], ensure_ascii=False), flush=True)
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
