"""Score the frozen compound corpus with parse_plan only. No model calls."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ollama_runner.inventory.registry import DeviceRegistry
from ollama_runner.nlu.structure import parse_plan


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "compound_cases.json"
RESULTS_PATH = ROOT / "tests" / "compound_results.json"


def _status(plan) -> str:
    if plan.fully_parsed and plan.commands:
        return "fully_parsed"
    if plan.commands and plan.unresolved_spans:
        return "partially_parsed"
    return "unhandled"


def _got(command) -> dict[str, str | None]:
    target = command.target
    return {
        "intent": command.intent,
        "device_type": target.device_type,
        "area": target.area,
        "owner": target.owner,
    }


def _same(expected: dict, actual: dict) -> bool:
    for key in ("intent", "device_type", "area", "owner"):
        if key in expected and (expected.get(key) or None) != (actual.get(key) or None):
            return False
    return True


def main() -> None:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    registry = DeviceRegistry.from_static()
    rows = []
    false_extra = 0
    missing = 0
    for case in payload["cases"]:
        plan = parse_plan(case["text"], registry)
        status = _status(plan)
        actual = [_got(command) for command in plan.commands]
        expected = case["commands"]
        matched = 0
        used = [False] * len(actual)
        for want in expected:
            for index, got in enumerate(actual):
                if used[index] or not _same(want, got):
                    continue
                used[index] = True
                matched += 1
                break
        extra = len(actual) - matched
        absent = len(expected) - matched
        false_extra += extra
        missing += absent
        correct = status == case["expected_status"] and extra == 0 and absent == 0
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "text": case["text"],
            "expected_status": case["expected_status"],
            "status": status,
            "expected": expected,
            "actual": actual,
            "unresolved": list(plan.unresolved_spans),
            "correct": correct,
            "extra": extra,
            "missing": absent,
        })

    print(f"total: {len(rows)}")
    print("fully parsed:", sum(row["status"] == "fully_parsed" for row in rows))
    print("partially parsed:", sum(row["status"] == "partially_parsed" for row in rows))
    print("unhandled:", sum(row["status"] == "unhandled" for row in rows))
    print("fully correct:", sum(row["correct"] for row in rows))
    print("incorrect:", sum(not row["correct"] for row in rows))
    print("false extra commands:", false_extra)
    print("missing commands:", missing)
    print("\nINCORRECT")
    for row in rows:
        if row["correct"]:
            continue
        print(f"{row['id']} {row['expected_status']} -> {row['status']} extra={row['extra']} missing={row['missing']}")
        print(f"  {row['text']}")
        print(f"  expected {row['expected']}")
        print(f"  actual   {row['actual']}")
        if row["unresolved"]:
            print(f"  unresolved {row['unresolved']}")
    by_cat = Counter(row["category"] for row in rows if not row["correct"])
    print("\nincorrect by category", dict(by_cat))
    RESULTS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
