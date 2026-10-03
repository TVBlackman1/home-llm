"""Score the frozen challenge corpus with parse_deterministic only.

Does not call Ministral and does not change the grammar.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ollama_runner.inventory.registry import DeviceRegistry, fold
from ollama_runner.nlu.parse import parse_deterministic


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "challenge_cases.json"
RESULTS_PATH = ROOT / "tests" / "challenge_results.json"
REJECT = "NEEDS_CONTEXT"

_TEMPLATE_SUBS = (
    (r"[«»\"'()—–\-.,!?:;]+", " "),
    (r"\b(кварис\w*|норвен\w*|элидар\w*)\b", " TITLE "),
    (r"\b(маш\w*|антон\w*|андр\w*|артем\w*|марин\w*|мам\w*|пап\w*)\b", " OWNER "),
    (r"\b(кухн\w*|спальн\w*|гостин\w*|коридор\w*|ванн\w*|кабинет\w*|офис\w*|балкон\w*|детск\w*)\b", " AREA "),
    (r"\b(телевизор\w*|свет\w*|ламп\w*|колонк\w*|торшер\w*|светильник\w*|монитор\w*|наушник\w*|проектор\w*|пылесос\w*|ночник\w*)\b", " DEV "),
    (r"\b(перв\w+|втор\w+|трет\w+|четверт\w+|\d+|десять|пять|двадцать|тридцать|сорок|пятнадцать)\b", " N "),
)


def template_of(text: str) -> str:
    folded = fold(text)
    for pattern, replacement in _TEMPLATE_SUBS:
        folded = re.sub(pattern, replacement, folded)
    return re.sub(r"\s+", " ", folded).strip()


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{part / whole:.1%}"


def _row(case: dict, parsed) -> dict:
    predicted = parsed.intent if parsed.handled else None
    expected = case["expected_intent"]
    if expected == REJECT:
        correct = not parsed.handled
    else:
        correct = parsed.handled and predicted == expected
    false_handled = parsed.handled and not correct
    return {
        "id": case["id"],
        "category": case["category"],
        "origin": case["origin"],
        "text": case["text"],
        "expected_intent": expected,
        "handled": parsed.handled,
        "predicted_intent": predicted,
        "correct": correct,
        "false_handled": false_handled,
        "reason": parsed.reason if not parsed.handled else (parsed.evidence[0] if parsed.evidence else parsed.reason),
        "evidence": list(parsed.evidence),
        "note": case.get("note", ""),
        "template": template_of(case["text"]),
    }


def _bucket(rows: list[dict]) -> dict:
    n = len(rows)
    handled_rows = [row for row in rows if row["handled"]]
    correct_handled = [row for row in handled_rows if row["correct"]]
    false_rows = [row for row in rows if row["false_handled"]]
    return {
        "n": n,
        "handled": len(handled_rows),
        "unhandled": n - len(handled_rows),
        "correct_handled": len(correct_handled),
        "incorrect_handled": len(false_rows),
        "coverage": _pct(len(handled_rows), n),
        "handled_accuracy": _pct(len(correct_handled), len(handled_rows)),
        "false_handled": len(false_rows),
        "correct_rejection": sum(1 for row in rows if row["expected_intent"] == REJECT and not row["handled"]),
    }


def _print_bucket(title: str, stats: dict) -> None:
    print(title)
    print(f"  n: {stats['n']}")
    print(f"  handled: {stats['handled']}")
    print(f"  unhandled: {stats['unhandled']}")
    print(f"  coverage: {stats['coverage']}")
    print(f"  correct handled: {stats['correct_handled']}")
    print(f"  handled accuracy: {stats['handled_accuracy']}")
    print(f"  false-handled: {stats['false_handled']}")
    print(f"  correct rejection: {stats['correct_rejection']}")


def _residual(rows: list[dict]) -> None:
    missed = [row for row in rows if not row["handled"]]
    counts = Counter(row["expected_intent"] for row in missed)
    print(f"  unhandled: {len(missed)}")
    for intent, count in counts.most_common():
        print(f"    {intent}: {count}")
        samples = [row["text"] for row in missed if row["expected_intent"] == intent][:4]
        for text in samples:
            print(f"      - {text}")


def main() -> None:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    registry = DeviceRegistry.from_static()
    rows = [_row(case, parse_deterministic(case["text"], registry)) for case in payload["cases"]]

    templates = [row["template"] for row in rows]
    repeated = {template: count for template, count in Counter(templates).items() if count > 1}

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    print(f"cases: {len(rows)}")
    print(f"unique templates: {len(set(templates))} / {len(rows)}")
    print(f"repeated templates: {len(repeated)}")
    for template, count in sorted(repeated.items(), key=lambda item: -item[1]):
        print(f"  {count} {template}")

    order = ("parser_robustness", "semantic_paraphrase", "ambiguous", "adversarial")
    summary = {}
    for category in order:
        stats = _bucket(by_category[category])
        summary[category] = stats
        print()
        _print_bucket(category, stats)

    overall = _bucket(rows)
    actionable = _bucket(by_category["parser_robustness"] + by_category["semantic_paraphrase"])
    print()
    _print_bucket("overall", overall)
    print()
    _print_bucket("A+B only", actionable)

    print("\nRESIDUAL A parser-limited")
    _residual(by_category["parser_robustness"])
    print("\nRESIDUAL B paraphrase")
    _residual(by_category["semantic_paraphrase"])

    print("\nFALSE-HANDLED")
    for category in order:
        bad = [row for row in by_category[category] if row["false_handled"]]
        print(f"\n{category}: {len(bad)}")
        for row in bad:
            print(f"  {row['id']} expected={row['expected_intent']} predicted={row['predicted_intent']} evidence={row['evidence']}")
            print(f"    {row['text']}")

    print("\nB HANDLED CORRECT sample")
    good_b = [row for row in by_category["semantic_paraphrase"] if row["handled"] and row["correct"]]
    print(f"  {len(good_b)}")
    for row in good_b:
        print(f"  {row['id']} {row['predicted_intent']} | {row['text']}")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "cases": rows,
                "summary": summary,
                "overall": overall,
                "actionable": actionable,
                "unique_templates": len(set(templates)),
                "repeated_templates": repeated,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
