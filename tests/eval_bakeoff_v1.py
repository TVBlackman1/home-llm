"""One frozen production-path run per model. Prompt and schema stay identical."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

from ollama_runner.factory import semantic_pipeline
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.sinks.expect import Expect


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "bakeoff_v1"
PROMPT = ROOT / "src" / "ollama_runner" / "prompts" / "semantic.md"
HIGH = ("B-03", "B-12", "B-18", "B-20", "B-38")
PREVIOUS_WRONG = (
    "B-03", "B-12", "B-14", "B-18", "B-20", "B-38", "B-45", "B-48", "B-49",
    "B-55", "B-61", "B-62", "B-68", "B-79", "B-80", "B-81", "B-82", "B-92",
)
FALSE_REJECTS = ("B-16", "B-70", "B-71", "B-73", "B-76", "B-90", "B-91")
STABILITY_IDS = PREVIOUS_WRONG + FALSE_REJECTS  # adversarial ids added from the corpus


def _base():
    path = ROOT / "tests" / "eval_intent_authoritative_v1.py"
    spec = importlib.util.spec_from_file_location("intent_auth_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resident() -> list[dict]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as response:
            return json.loads(response.read().decode()).get("models") or []
    except Exception as error:  # noqa: BLE001
        return [{"error": str(error)}]


def _watch(pipeline, samples: list[dict]) -> None:
    client = pipeline._nlu.client
    original = client.post

    def post(url, json=None, **kwargs):  # noqa: A002
        response = original(url, json=json, **kwargs)
        if response.status_code == 200:
            body = response.json()
            eval_count = body.get("eval_count") or 0
            eval_ns = body.get("eval_duration") or 0
            samples.append({
                "load_s": (body.get("load_duration") or 0) / 1e9,
                "prompt_eval_s": (body.get("prompt_eval_duration") or 0) / 1e9,
                "prompt_eval_count": body.get("prompt_eval_count") or 0,
                "eval_s": eval_ns / 1e9,
                "eval_count": eval_count,
                "tokens_per_s": (eval_count / (eval_ns / 1e9)) if eval_count and eval_ns else None,
            })
        return response

    client.post = post


def _challenge_row(base, pipeline, registry, mapping, case: dict) -> dict:
    calls = pipeline._nlu.calls
    pipeline.run(case["text"], base._Sink())
    obs = pipeline.observations[-1]
    expected = mapping[case["id"]]
    predicted = "command" if obs["handled"] else obs["outcome"]
    final_intent = obs["final"]["intent"] if obs["final"] else None
    raw_intent = obs["llm"]["intent"] if obs["llm"] else None
    evidence = base._text_slots(case["text"], registry)
    intent_correct = predicted == "command" and final_intent == expected["intent"]
    outcome_correct = predicted == expected["outcome"] and (
        expected["outcome"] != "command" or intent_correct
    )
    return {
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
        "blocked": base._blocked(obs["llm"], obs["final"], evidence) if predicted == "command" else [],
        "resolver_status": obs["resolver_status"],
        "executed": predicted == "command",
        "outcome_correct": predicted == expected["outcome"],
        "intent_correct": intent_correct,
        "pipeline_correct": outcome_correct,
        "latency_s": obs["total_s"],
        "fallback_s": obs["fallback_s"],
        "llm_calls": pipeline._nlu.calls - calls,
    }


def _pack(row: dict) -> dict:
    executable = row["predicted_outcome"] == "command" and row["resolver_status"] == "resolved"
    return {
        "id": row["id"],
        "text": row["text"],
        "expected_outcome": row["expected_outcome"],
        "expected_intent": row["expected_intent"],
        "outcome": row["predicted_outcome"],
        "intent": row["final_intent"],
        "resolver": row["resolver_status"],
        "deterministic": row["deterministic_handled"],
        "executable": executable,
        "correct": row["pipeline_correct"],
    }


def _focus(rows: list[dict]) -> dict:
    by_id = {row["id"]: row for row in rows}
    adversarial = [row for row in rows if row["category"] == "adversarial"]
    wrong = [by_id[case_id] for case_id in PREVIOUS_WRONG]
    rejects = [by_id[case_id] for case_id in FALSE_REJECTS]
    return {
        "high": [_pack(by_id[case_id]) for case_id in HIGH],
        "b33": _pack(by_id["B-33"]),
        "previous_18": {
            "correct": sum(row["pipeline_correct"] for row in wrong),
            "wrong_intent": sum(
                row["predicted_outcome"] == "command" and not row["pipeline_correct"] for row in wrong
            ),
            "false_reject": sum(row["predicted_outcome"] != "command" for row in wrong),
            "rows": [_pack(row) for row in wrong],
        },
        "false_rejects": [_pack(row) for row in rejects],
        "adversarial": {
            "n": len(adversarial),
            "not_command": sum(row["predicted_outcome"] == "not_command" for row in adversarial),
            "needs_context": sum(row["predicted_outcome"] == "needs_context" for row in adversarial),
            "command_blocked": sum(
                row["predicted_outcome"] == "command" and row["resolver_status"] != "resolved"
                for row in adversarial
            ),
            "command_executable": sum(
                row["predicted_outcome"] == "command" and row["resolver_status"] == "resolved"
                for row in adversarial
            ),
            "rows": [_pack(row) for row in adversarial],
        },
    }


def _safety(rows: list[dict]) -> dict:
    llm = []
    deterministic = []
    for row in rows:
        if row["expected_outcome"] != "not_command" or row["predicted_outcome"] != "command":
            continue
        item = _pack(row)
        if row["deterministic_handled"]:
            deterministic.append(item)
        else:
            llm.append(item)
    return {
        "llm_false_command": llm,
        "llm_blocked": [row for row in llm if not row["executable"]],
        "llm_executable": [row for row in llm if row["executable"]],
        "deterministic_false_command": deterministic,
    }


def _signature(row: dict) -> tuple:
    return (row["predicted_outcome"], row["final_intent"])


def main() -> None:
    model = sys.argv[1]
    base = _base()
    prompt_text = PROMPT.read_text(encoding="utf-8")
    if "## Различения" in prompt_text:
        raise SystemExit("prompt still contains semantic-prompt-v2 section")
    prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
    registry = DeviceRegistry.from_static()
    mapping = {
        row["id"]: row
        for row in json.loads((ROOT / "tests" / "reject_outcome_mapping.json").read_text(encoding="utf-8"))["cases"]
    }
    pipeline = semantic_pipeline(model=model)
    samples: list[dict] = []
    _watch(pipeline, samples)
    before = pipeline._nlu.calls
    started = time.perf_counter()
    pipeline.warmup()
    warmup_s = time.perf_counter() - started
    warmup_calls = pipeline._nlu.calls - before
    resident = _resident()
    print(f"warmup {warmup_s:.2f}s resident={resident}", flush=True)

    regression_rows = []
    for case in base._load_regression():
        calls = pipeline._nlu.calls
        expected = case.get("semantic_expected", case["expected"])
        result = pipeline.run(case["text"], Expect(expected))
        obs = pipeline.observations[-1]
        resolution = case.get("resolution")
        ok = base._resolution_matches(result, resolution) if resolution else result.ok
        regression_rows.append({
            "id": case["id"],
            "ok": ok,
            "fallback": not obs["handled"],
            "llm_calls": pipeline._nlu.calls - calls,
        })
    print(
        f"regression {sum(row['ok'] for row in regression_rows)}/{len(regression_rows)} "
        f"fallback {sum(row['fallback'] for row in regression_rows)}",
        flush=True,
    )

    challenge = json.loads((ROOT / "tests" / "challenge_cases.json").read_text(encoding="utf-8"))
    challenge_rows = []
    for index, case in enumerate(challenge["cases"], start=1):
        row = _challenge_row(base, pipeline, registry, mapping, case)
        challenge_rows.append(row)
        if index % 20 == 0 or row["fallback_called"]:
            print(
                f"[{index:03d}] {case['id']} {row['predicted_outcome']} "
                f"{'ok' if row['pipeline_correct'] else 'miss'} {row['fallback_s']:.2f}s",
                flush=True,
            )

    compound = json.loads((ROOT / "tests" / "compound_cases.json").read_text(encoding="utf-8"))
    compound_rows = []
    for case in compound["cases"]:
        calls = pipeline._nlu.calls
        pipeline.run(case["text"], base._Sink())
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
        _, structural_extra, structural_missing = base._match_commands(case["commands"], structural_actual)
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
        _, extra, missing = base._match_commands(case["commands"], executed)
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
            "final_intent": obs["final"]["intent"] if obs["final"] else None,
            "resolver_status": obs["resolver_status"],
            "executed": executed,
            "executed_extra": extra,
            "executed_missing": missing,
            "llm_calls": pipeline._nlu.calls - calls,
            "fallback_s": obs["fallback_s"],
            "latency_s": obs["total_s"],
        })
        print(f"compound {case['id']} {predicted} struct={'ok' if structural_correct else 'miss'}", flush=True)

    by_id = {row["id"]: row for row in challenge_rows}
    adversarial_ids = tuple(row["id"] for row in challenge_rows if row["category"] == "adversarial")
    stability_ids = STABILITY_IDS + adversarial_ids
    passes = [[_signature(by_id[case_id]) for case_id in stability_ids]]
    for pass_index in (2, 3):
        print(f"stability pass {pass_index}", flush=True)
        current = []
        for case_id in stability_ids:
            row = _challenge_row(base, pipeline, registry, mapping, {
                "id": case_id,
                "text": by_id[case_id]["text"],
                "category": by_id[case_id]["category"],
            })
            current.append(_signature(row))
            print(f"  {case_id} {row['predicted_outcome']} {row['final_intent']}", flush=True)
        passes.append(current)
    unstable = []
    for index, case_id in enumerate(stability_ids):
        seen = [item[index] for item in passes]
        if len(set(seen)) != 1:
            unstable.append({"id": case_id, "text": by_id[case_id]["text"], "runs": seen})

    summary = base._summarize(
        regression_rows,
        challenge_rows,
        compound_rows,
        warmup_s,
        warmup_calls,
        pipeline._nlu.calls,
        resident,
    )
    summary["model"] = model
    summary["prompt_sha256"] = prompt_sha
    summary["prompt_chars"] = len(prompt_text)
    summary["focus"] = _focus(challenge_rows)
    summary["safety_split"] = _safety(challenge_rows)
    summary["stability"] = {
        "n": len(stability_ids),
        "runs": 3,
        "stable": len(stability_ids) - len(unstable),
        "unstable": unstable,
    }
    token_rates = [item["tokens_per_s"] for item in samples if item["tokens_per_s"]]
    summary["generation"] = {
        "calls": len(samples),
        "cold_load_s": samples[0]["load_s"] if samples else None,
        "cold_request_s": warmup_s,
        "prompt_eval_s_p50": base._percentile([item["prompt_eval_s"] for item in samples], 50) if samples else None,
        "tokens_per_s_mean": (sum(token_rates) / len(token_rates)) if token_rates else None,
    }
    summary["loaded"] = [
        {
            "name": item.get("name"),
            "size": item.get("size"),
            "size_vram": item.get("size_vram"),
            "processor": (item.get("details") or {}).get("processor") if isinstance(item.get("details"), dict) else None,
        }
        for item in resident
        if isinstance(item, dict)
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = model.replace(":", "_").replace("/", "_")
    out = OUT_DIR / f"{slug}.json"
    out.write_text(
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
    print(f"wrote {out}", flush=True)
    pipeline._nlu.close()


if __name__ == "__main__":
    main()
