"""Offline reject-semantics prompt experiment.

Does not modify the production prompt, parser, merge, resolver, or datasets.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ollama_runner.inventory.registry import DeviceRegistry, fold
from ollama_runner.nlu.backend import LLMNLUBackend
from ollama_runner.nlu.slots import read_slots, spoken_mention
from ollama_runner.nlu.structure import parse_plan
from ollama_runner.resolve.capability import CapabilityResolver
from ollama_runner.semantic import CommandOutcome, RequestContext
from ollama_runner.semantic_pipeline import SemanticPipeline
from ollama_runner.settings import get_settings
from ollama_runner.sinks.expect import Expect
from ollama_runner.skills.book import Executor
from ollama_runner.types import Command, Result


ROOT = Path(__file__).resolve().parents[1]
PROD_PROMPT = ROOT / "src/ollama_runner/prompts/semantic.md"
PROMPTS = {
    "production": PROD_PROMPT,
    "v2a": ROOT / "tests/prompts/reject_semantics_v2a.md",
    "v2b": ROOT / "tests/prompts/reject_semantics_v2b.md",
}
RUNS = ROOT / "tests/reject_semantics_v2_runs.jsonl"
SUMMARY = ROOT / "tests/reject_semantics_v2_summary.json"
REJECT = "NEEDS_CONTEXT"
REPEATS = 3

FOCUSED = [
    ("D-01", "Свет в конце тоннеля меня не включает", "reject"),
    ("D-03", "Телевизор в новостях опять показывает политику", "reject"),
    ("D-11", "Не включай телевизор", "reject"),
    ("D-25", "Сделай вид, что свет не горит", "reject"),
    ("obs-on", "Свет горит", "observation"),
    ("obs-off", "Свет не горит", "observation"),
    ("obs-tv", "Телевизор показывает политику", "observation"),
    ("cmd-on-light", "Включи свет", "explicit"),
    ("cmd-off-light", "Выключи свет", "explicit"),
    ("cmd-on-tv", "Включи телевизор", "explicit"),
    ("cmd-off-tv", "Выключи телевизор", "explicit"),
    ("cmd-film", "Поставь фильм Интерстеллар", "explicit"),
    ("cmd-dim", "Сделай свет потемнее", "explicit"),
    ("B-13", "Свет в ванной больше не нужен", "indirect"),
    ("B-22", "Света не хватает в ванной", "indirect"),
]

INDIRECT_IDS = (
    "B-03", "B-05", "B-12", "B-13", "B-19", "B-20", "B-22", "B-23",
    "B-25", "B-27", "B-32", "B-33", "B-35", "B-39", "B-41", "B-45",
    "B-46", "B-49",
)


class Accept:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class RecordingNLU:
    def __init__(self, inner: LLMNLUBackend) -> None:
        self.inner = inner
        self.last: Any = None

    def parse(self, text: str, context: RequestContext | None = None):
        self.last = self.inner.parse(text, context)
        return self.last


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(cases: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for case in cases:
        if case["text"] in seen:
            continue
        seen.add(case["text"])
        unique.append(case)
    return unique


def _regression_cases() -> list[dict]:
    embed_path = ROOT / "tests/embed_cases.json"
    embed = _load_json(embed_path) if embed_path.exists() else []
    return _dedupe(embed + _load_json(ROOT / "tests/cases.json"))


def _resolution_matches(result: Result, resolution: dict | None) -> bool:
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


def _view(command) -> dict | None:
    if command is None:
        return None
    target = command.target
    return {
        "intent": command.intent,
        "device_type": target.device_type or "",
        "mention": target.mention or "",
        "owner": target.owner or "",
        "area": target.area or "",
        "ordinal": target.ordinal or 0,
        "explicit": bool(target.explicit),
        "content": command.arguments.get("content") or "",
        "value": command.arguments.get("value") or "",
    }


def _outcome_view(outcome) -> dict:
    if isinstance(outcome, CommandOutcome):
        viewed = _view(outcome.command) or {}
        return {"outcome": "command", **viewed}
    kind = getattr(outcome, "kind", None) or "missing"
    return {
        "outcome": kind,
        "intent": "",
        "device_type": "",
        "mention": "",
        "owner": "",
        "area": "",
        "ordinal": 0,
        "explicit": False,
        "content": "",
        "value": "",
    }


def _same(value, canonical, raw, text: str) -> bool:
    if value in (None, "", 0):
        return True
    if isinstance(value, int):
        return value == (canonical or 0)
    folded = fold(str(value))
    if folded and folded in fold(text):
        return True
    if canonical and fold(str(canonical)) == folded:
        return True
    if raw and fold(str(raw)) == folded:
        return True
    return False


def _hallucinations(llm: dict, final: dict | None, text: str, registry: DeviceRegistry) -> dict:
    if llm.get("outcome") != "command":
        return {"produced": [], "survived": []}
    slots = read_slots(text, registry)
    mention = spoken_mention(text, registry)
    pairs = (
        ("device_type", llm.get("device_type") or "", slots.device_type, slots.raw_device_type),
        ("owner", llm.get("owner") or "", slots.owner, slots.raw_owner),
        ("area", llm.get("area") or "", slots.area, slots.raw_area),
        ("mention", llm.get("mention") or "", mention, None),
        ("ordinal", llm.get("ordinal") or 0, slots.ordinal or 0, None),
    )
    produced = []
    survived = []
    final = final or {}
    for name, value, canonical, raw in pairs:
        if _same(value, canonical, raw, text):
            continue
        produced.append({"field": name, "llm": value, "text": canonical})
        if final.get(name) == value:
            survived.append(name)
    return {"produced": produced, "survived": survived}


def _pipeline(prompt: Path, registry: DeviceRegistry) -> tuple[SemanticPipeline, RecordingNLU]:
    settings = get_settings()
    backend = LLMNLUBackend(
        model=settings.ollama_model,
        prompt_path=prompt,
        base_url=settings.ollama_url,
        temperature=settings.ollama_temperature,
        keep_alive=settings.ollama_keep_alive,
        timeout=settings.ollama_timeout,
        think=False,
    )
    recording = RecordingNLU(backend)
    pipeline = SemanticPipeline(
        recording,
        CapabilityResolver(registry),
        Executor(registry),
        registry,
    )
    return pipeline, recording


def _append(row: dict) -> None:
    with RUNS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done(variant: str, split: str) -> int:
    if not RUNS.exists():
        return 0
    count = 0
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("variant") == variant and row.get("split") == split:
            count += 1
    return count


def _run_focused(variant: str, pipeline: SemanticPipeline, recording: RecordingNLU) -> None:
    if _done(variant, "focused") >= len(FOCUSED) * REPEATS:
        print(f"{variant} focused already complete", flush=True)
        return
    for case_id, text, role in FOCUSED:
        votes = []
        for attempt in range(1, REPEATS + 1):
            recording.last = None
            outcome = recording.parse(text, RequestContext())
            view = _outcome_view(outcome)
            votes.append(view["outcome"] if view["outcome"] != "command" else f"command:{view['intent']}")
            _append({
                "variant": variant,
                "split": "focused",
                "id": case_id,
                "role": role,
                "text": text,
                "attempt": attempt,
                "llm": view,
            })
        print(f"{variant} {case_id} {Counter(votes)} | {text}", flush=True)


def _run_regression(variant: str, pipeline: SemanticPipeline, recording: RecordingNLU, cases: list[dict]) -> None:
    expected_rows = len(cases) * REPEATS
    if _done(variant, "regression") >= expected_rows:
        print(f"{variant} regression already complete", flush=True)
        return
    before = recording.inner.calls
    for case in cases:
        for attempt in range(1, REPEATS + 1):
            before_obs = len(pipeline.observations)
            expected = case.get("semantic_expected", case["expected"])
            result = pipeline.run(case["text"], Expect(expected))
            called = len(pipeline.observations) > before_obs and not pipeline.observations[-1]["handled"]
            resolution = case.get("resolution")
            ok = _resolution_matches(result, resolution) if resolution else result.ok
            payload = result.payload or {}
            _append({
                "variant": variant,
                "split": "regression",
                "id": case["id"],
                "text": case["text"],
                "attempt": attempt,
                "ok": ok,
                "called_llm": called,
                "outcome": payload.get("outcome") or payload.get("status"),
                "error": None if ok else (result.error or "")[:240],
            })
            pipeline.observations.clear()
            pipeline.traces.clear()
    print(
        f"{variant} regression rows={expected_rows} llm_calls_delta={recording.inner.calls - before}",
        flush=True,
    )


def _run_challenge(
    variant: str,
    pipeline: SemanticPipeline,
    recording: RecordingNLU,
    cases: list[dict],
    registry: DeviceRegistry,
) -> None:
    expected_rows = len(cases) * REPEATS
    have = _done(variant, "challenge")
    if have >= expected_rows:
        print(f"{variant} challenge already complete", flush=True)
        return
    start_case = have // REPEATS
    for index, case in enumerate(cases):
        if index < start_case:
            continue
        for attempt in range(1, REPEATS + 1):
            if index == start_case and (index * REPEATS + attempt) <= have:
                continue
            recording.last = None
            result = pipeline.run(case["text"], Accept())
            obs = pipeline.observations[-1]
            llm = _outcome_view(recording.last) if recording.last is not None else {
                "outcome": "deterministic",
                "intent": obs.get("final", {}).get("intent") if obs.get("final") else "",
            }
            final = obs.get("final")
            halls = _hallucinations(llm, final, case["text"], registry) if obs.get("outcome") == "command" else {
                "produced": [],
                "survived": [],
            }
            _append({
                "variant": variant,
                "split": "challenge",
                "id": case["id"],
                "category": case["category"],
                "text": case["text"],
                "expected": case["expected_intent"],
                "attempt": attempt,
                "handled": obs["handled"],
                "llm": llm,
                "final_intent": (final or {}).get("intent") or "",
                "resolver_status": obs.get("resolver_status"),
                "resolver_reason": obs.get("resolver_reason"),
                "hallucinations": halls,
            })
            pipeline.observations.clear()
            pipeline.traces.clear()
        if (index + 1) % 20 == 0:
            print(f"{variant} challenge {index + 1}/{len(cases)}", flush=True)
    print(f"{variant} challenge done", flush=True)


def _signature(row: dict) -> str:
    llm = row.get("llm") or {}
    if row.get("handled"):
        return f"det:{row.get('final_intent')}"
    outcome = llm.get("outcome")
    if outcome == "command":
        return f"command:{llm.get('intent')}"
    return outcome or "missing"


def _majority(items: list[str]) -> str:
    return Counter(items).most_common(1)[0][0]


def _summarize(cases: list[dict]) -> dict:
    rows = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_case = {case["id"]: case for case in cases}
    summary = {}
    for variant in PROMPTS:
        challenge = [row for row in rows if row["variant"] == variant and row["split"] == "challenge"]
        regression = [row for row in rows if row["variant"] == variant and row["split"] == "regression"]
        focused = [row for row in rows if row["variant"] == variant and row["split"] == "focused"]
        grouped: dict[str, list[dict]] = {}
        for row in challenge:
            grouped.setdefault(row["id"], []).append(row)
        correct = false_cmd = exec_cmd = false_reject = wrong = unstable = 0
        false_c, false_d = [], []
        exec_ids = []
        remaining = []
        rejects = []
        wrongs = []
        unstable_ids = []
        fallback = fallback_correct = 0
        produced_cases = survived_cases = reached = 0
        det_false = []
        indirect = {}
        for case_id, runs in grouped.items():
            case = by_case[case_id]
            expected = case["expected_intent"]
            sigs = [_signature(row) for row in runs]
            if len(set(sigs)) > 1:
                unstable += 1
                unstable_ids.append(case_id)
            labels = []
            for row in runs:
                llm = row["llm"]
                outcome = llm.get("outcome")
                intent = row["final_intent"] or llm.get("intent") or ""
                if expected == REJECT:
                    if row["handled"]:
                        labels.append("det_false")
                    elif outcome == "command":
                        executable = row.get("resolver_status") == "resolved"
                        labels.append("exec" if executable else "false_cmd")
                    else:
                        labels.append("reject_ok")
                else:
                    if intent == expected and (row["handled"] or outcome == "command"):
                        labels.append("correct")
                    elif outcome in {"not_command", "needs_context"} or intent == "":
                        labels.append("false_reject")
                    else:
                        labels.append("wrong")
                if row["llm"].get("outcome") == "command" or not row["handled"]:
                    if not row["handled"]:
                        pass
                if not row["handled"]:
                    pass
                produced = row.get("hallucinations", {}).get("produced") or []
                survived = row.get("hallucinations", {}).get("survived") or []
                if produced:
                    produced_cases += 1
                if survived:
                    survived_cases += 1
                    reached += 1
            vote = _majority(labels)
            called = any(not row["handled"] for row in runs)
            if called:
                fallback += 1
                if vote == "correct" or (expected == REJECT and vote == "reject_ok"):
                    fallback_correct += 1
            if vote == "correct" or vote == "reject_ok":
                correct += 1
            elif vote == "false_reject":
                false_reject += 1
                rejects.append({"id": case_id, "text": case["text"], "expected": expected, "votes": sigs})
            elif vote == "wrong":
                wrong += 1
                wrongs.append({"id": case_id, "text": case["text"], "expected": expected, "votes": sigs})
            elif vote in {"false_cmd", "exec", "det_false"}:
                pass
            if expected == REJECT and vote in {"false_cmd", "exec"}:
                false_cmd += 1
                if case["category"] == "ambiguous":
                    bucket = false_c
                elif case["category"] == "adversarial":
                    bucket = false_d
                else:
                    bucket = false_d
                bucket.append(case_id)
                executable = vote == "exec"
                if executable:
                    exec_cmd += 1
                    exec_ids.append(case_id)
                remaining.append({
                    "id": case_id,
                    "category": case["category"],
                    "text": case["text"],
                    "votes": sigs,
                    "executable": executable,
                    "resolver": [row.get("resolver_status") for row in runs],
                })
            if expected == REJECT and vote == "det_false":
                det_false.append(case_id)
                correct -= 0
            if case_id in INDIRECT_IDS:
                indirect[case_id] = {"text": case["text"], "expected": expected, "votes": sigs, "label": vote}
        reg_grouped: dict[str, list[dict]] = {}
        for row in regression:
            reg_grouped.setdefault(row["id"], []).append(row)
        reg_correct = reg_false = reg_wrong = reg_unstable = 0
        reg_llm = 0
        for case_id, runs in reg_grouped.items():
            oks = [row["ok"] for row in runs]
            if any(row["called_llm"] for row in runs):
                reg_llm += 1
            if len(set(row["outcome"] for row in runs)) > 1 or len(set(oks)) > 1:
                reg_unstable += 1
            if all(oks):
                reg_correct += 1
            elif any((row.get("outcome") in {"not_command", "needs_context"}) for row in runs):
                reg_false += 1
            else:
                reg_wrong += 1
        focused_out = {}
        for row in focused:
            focused_out.setdefault(row["id"], []).append(
                row["llm"]["outcome"] if row["llm"]["outcome"] != "command" else f"command:{row['llm']['intent']}"
            )
        summary[variant] = {
            "regression": {
                "correct": reg_correct,
                "n": len(reg_grouped),
                "false_rejects": reg_false,
                "wrong": reg_wrong,
                "unstable": reg_unstable,
                "llm_cases": reg_llm,
            },
            "challenge": {
                "correct": correct,
                "n": len(grouped),
                "fallback": fallback,
                "fallback_correct": fallback_correct,
                "false_command": false_cmd,
                "false_command_c": false_c,
                "false_command_d": false_d,
                "executable_false": exec_cmd,
                "executable_ids": exec_ids,
                "non_executable_false": false_cmd - exec_cmd,
                "false_rejects": false_reject,
                "wrong_intents": wrong,
                "unstable": unstable,
                "unstable_ids": unstable_ids,
                "deterministic_false": det_false,
                "hallucinations_produced_runs": produced_cases,
                "hallucinations_survived_runs": survived_cases,
                "hallucinations_reached_resolver_runs": reached,
            },
            "remaining": remaining,
            "false_rejects": rejects,
            "wrong_intents": wrongs,
            "indirect": indirect,
            "focused": {key: Counter(value) for key, value in focused_out.items()},
        }
    return summary


def main() -> None:
    registry = DeviceRegistry.from_static()
    regression = _regression_cases()
    challenge = _load_json(ROOT / "tests/challenge_cases.json")["cases"]
    print(f"regression {len(regression)} challenge {len(challenge)}", flush=True)
    only = sys.argv[1:] or list(PROMPTS)
    for variant in only:
        prompt = PROMPTS[variant]
        print(f"=== {variant} {prompt} ===", flush=True)
        pipeline, recording = _pipeline(prompt, registry)
        pipeline.warmup()
        _run_focused(variant, pipeline, recording)
        _run_regression(variant, pipeline, recording, regression)
        _run_challenge(variant, pipeline, recording, challenge, registry)
        recording.inner.close()
    if set(only) == set(PROMPTS):
        summary = _summarize(challenge)
        printable = {}
        for variant, payload in summary.items():
            printable[variant] = {
                "regression": payload["regression"],
                "challenge": payload["challenge"],
                "remaining": payload["remaining"],
                "false_rejects": payload["false_rejects"],
                "wrong_intents": payload["wrong_intents"],
                "indirect": payload["indirect"],
                "focused": {key: dict(value) for key, value in payload["focused"].items()},
            }
        SUMMARY.write_text(json.dumps(printable, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v["challenge"] | {"regression": v["regression"]} for k, v in printable.items()}, ensure_ascii=False, indent=2))
        print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
