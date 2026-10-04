"""Audit-only. Runs the challenge corpus through the production pipeline.

Does not change the prompt, parser, merge, resolver, or expected labels.
"""

from __future__ import annotations

import json
from pathlib import Path

from ollama_runner.factory import semantic_pipeline
from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.parse import parse_deterministic
from ollama_runner.nlu.slots import read_slots, spoken_mention
from ollama_runner.semantic import CommandOutcome
from ollama_runner.types import Command, Result


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "challenge_cases.json"
OUT = ROOT / "tests" / "false_command_audit_v1.json"
REJECT = "NEEDS_CONTEXT"
TARGET_FIELDS = ("device_type", "mention", "owner", "area", "ordinal")


class Accept:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class RecordingNLU:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.last = None

    def parse(self, text: str, context):
        self.last = self.inner.parse(text, context)
        return self.last

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()


def _blank(value) -> bool:
    return value is None or value == "" or value == 0


def _view_target(command) -> dict:
    target = command.target
    return {
        "intent": command.intent,
        "device_type": target.device_type,
        "mention": target.mention,
        "owner": target.owner,
        "area": target.area,
        "ordinal": target.ordinal,
        "explicit": target.explicit,
        "content": command.arguments.get("content"),
        "value": command.arguments.get("value"),
    }


def _text_facts(text: str, registry: DeviceRegistry) -> dict:
    slots = read_slots(text, registry)
    return {
        "device_type": slots.device_type,
        "mention": spoken_mention(text, registry),
        "owner": slots.owner,
        "area": slots.area,
        "ordinal": slots.ordinal,
    }


def _hallucinated(raw: dict | None, facts: dict) -> list[str]:
    if raw is None:
        return []
    found = []
    for field in TARGET_FIELDS:
        value = raw.get(field)
        if _blank(value):
            continue
        if value != facts.get(field):
            found.append(field)
    return found


def main() -> None:
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    registry = DeviceRegistry.from_static()
    pipeline = semantic_pipeline()
    recorder = RecordingNLU(pipeline._nlu)
    pipeline._nlu = recorder
    pipeline.warmup()

    rows = []
    for index, case in enumerate(payload["cases"], start=1):
        text = case["text"]
        deterministic = parse_deterministic(text, registry)
        result = pipeline.run(text, Accept())
        observed = pipeline.observations[-1]
        facts = _text_facts(text, registry)
        raw = None
        if isinstance(recorder.last, CommandOutcome):
            raw = _view_target(recorder.last.command)
        final = observed.get("final")
        produced = _hallucinated(raw, facts)
        survived = [field for field in produced if final and final.get(field) == raw.get(field)]
        payload_out = result.payload or {}
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "text": text,
            "expected": case["expected_intent"],
            "note": case.get("note", ""),
            "deterministic_handled": deterministic.handled,
            "deterministic_intent": deterministic.intent,
            "deterministic_content": deterministic.content,
            "deterministic_value": deterministic.value,
            "facts": facts,
            "outcome": observed.get("outcome"),
            "llm": raw,
            "merge_notes": observed.get("merge_notes"),
            "final": final,
            "resolver_status": observed.get("resolver_status") or payload_out.get("status"),
            "resolver_reason": observed.get("resolver_reason") or payload_out.get("reason"),
            "trace": payload_out.get("trace"),
            "ha_action": payload_out.get("ha_action"),
            "execution_target": payload_out.get("execution_target"),
            "ok": result.ok,
            "hallucinations_produced": produced,
            "hallucinations_survived_merge": survived,
        })
        if index % 20 == 0 or index == len(payload["cases"]):
            print(f"scored {index}/{len(payload['cases'])}", flush=True)

    reject_rows = [row for row in rows if row["expected"] == REJECT]
    llm_false = [
        row for row in reject_rows
        if not row["deterministic_handled"] and row["outcome"] == "command"
    ]
    deterministic_false = [
        row for row in reject_rows
        if row["deterministic_handled"]
    ]
    executable = [row for row in llm_false if row["resolver_status"] == "resolved"]
    rejected = [row for row in llm_false if row["resolver_status"] != "resolved"]

    summary = {
        "challenge": len(rows),
        "reject_expected": len(reject_rows),
        "false_llm_command": len(llm_false),
        "executable_false_command": len(executable),
        "rejected_downstream": len(rejected),
        "deterministic_false_command": len(deterministic_false),
        "hallucinations_produced": sum(len(row["hallucinations_produced"]) for row in rows if row["outcome"] == "command"),
        "hallucinations_survived_merge": sum(len(row["hallucinations_survived_merge"]) for row in rows if row["outcome"] == "command"),
        "executable_ids": [row["id"] for row in executable],
        "rejected_ids": [row["id"] for row in rejected],
        "deterministic_false_ids": [row["id"] for row in deterministic_false],
    }
    OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    recorder.close()


if __name__ == "__main__":
    main()
