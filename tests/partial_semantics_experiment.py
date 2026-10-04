"""Experiment-only partial semantics and deterministic binder.

Not imported by the production pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "partial_semantics_cases.json"
OUT = ROOT / "tests" / "partial_binder_v1.json"

_ORDERED_DOMAINS = frozenset({"media.playlist", "media.episode"})
_RESTORE_ALLOW = {
    "media.next": "media.previous",
    "media.previous": "media.next",
}

_COUNT = {
    "один": 1,
    "одну": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "три": 3,
}
_DIRECTION = {
    "назад": "previous",
    "вперед": "forward",
    "дальше": "forward",
}


def _norm(text: str) -> str:
    folded = text.casefold().replace("ё", "е")
    folded = folded.replace("?", "").replace("!", "").replace(".", "")
    return re.sub(r"\s+", " ", folded).strip()


def parse_partial(text: str) -> dict | None:
    """Structural partials only. A concrete object or an explicit domain stays out."""

    folded = _norm(text)
    if " от " in f" {folded} ":
        return None

    restore = _restore_previous(folded)
    if restore is not None:
        return restore
    return _relative_step(folded)


def _restore_previous(folded: str) -> dict | None:
    patterns = (
        r"верни предыдущее",
        r"верни то, что (?:было|играло)(?: раньше| до этого| перед этим)?",
        r"верни как было(?: до этого| раньше)?",
        r"отмени последнее (?:изменение|действие)",
    )
    if any(re.fullmatch(pattern, folded) for pattern in patterns):
        return {
            "operation": "restore_previous",
            "referent": "previous_executed_action",
        }
    return None


def _relative_step(folded: str) -> dict | None:
    counted = re.fullmatch(
        r"(?:вернись на |перейди на |шагни на |на )?"
        r"(один|одну|одна|два|две|три)"
        r"(?: шаг| шага)?"
        r" (назад|вперед|дальше)",
        folded,
    )
    if counted:
        return _step(_COUNT[counted.group(1)], _DIRECTION[counted.group(2)])

    bare = re.fullmatch(r"(один|одну) назад", folded)
    if bare:
        return _step(1, "previous")

    if folded == "на шаг назад":
        return _step(1, "previous")

    if folded == "вернись на один шаг":
        return _step(1, "previous")
    return None


def _step(count: int, direction: str) -> dict:
    return {
        "operation": "relative_step",
        "direction": direction,
        "count": count,
        "domain": "unresolved",
    }


def bind(partial: dict | None, context: dict) -> dict:
    """Context never creates an operation. Incompatible context stays unresolved."""

    if partial is None:
        return {"status": "not_partial", "command": None}

    if partial["operation"] == "restore_previous":
        previous = context.get("previous_executed_semantic_action")
        intent = _RESTORE_ALLOW.get(previous or "")
        if intent is None:
            return {"status": "unresolved", "command": None, "reason": "incompatible_history"}
        return {"status": "bound", "command": {"intent": intent}, "reason": "undo_navigation"}

    if partial["operation"] == "relative_step":
        if partial["count"] != 1:
            return {"status": "unresolved", "command": None, "reason": "representation_limit"}
        domain = context.get("active_domain")
        if domain not in _ORDERED_DOMAINS:
            return {"status": "unresolved", "command": None, "reason": "incompatible_domain"}
        intent = "media.previous" if partial["direction"] == "previous" else "media.next"
        return {"status": "bound", "command": {"intent": intent}, "reason": "ordered_domain"}

    return {"status": "unresolved", "command": None, "reason": "unknown_operation"}


def _same(found: dict | None, expected: dict | None) -> bool:
    return found == expected


def evaluate() -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    parsed = []
    tp = fp = fn = tn = 0
    false_partial = []
    missed = []
    for case in cases:
        found = parse_partial(case["text"])
        want = case["expected_kind"] == "partial"
        hit = found is not None
        ok = hit == want and (not want or _same(found, case["expected_partial"]))
        if want and ok:
            tp += 1
        elif want and not ok:
            fn += 1
            missed.append({"id": case["id"], "text": case["text"], "found": found})
        elif hit:
            fp += 1
            false_partial.append({"id": case["id"], "text": case["text"], "found": found, "role": case["role"]})
        else:
            tn += 1
        parsed.append({"id": case["id"], "text": case["text"], "found": found, "ok": ok})

    adversarial_false = [row for row in false_partial if row["role"] == "adversarial_control"]
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)

    history = ("media.next", "media.previous", "brightness.set", "device.turn_off", "none")
    domains = ("media.playlist", "media.episode", "media.photos", "none")
    matrix = []
    wrong = []
    correct_bind = 0
    correct_unresolved = 0
    for case in cases:
        partial = parse_partial(case["text"])
        if partial is None:
            poisoned = bind(None, {
                "previous_executed_semantic_action": "media.next",
                "active_domain": "media.playlist",
            })
            if poisoned["command"] is not None:
                wrong.append({"id": case["id"], "text": case["text"], "poison": poisoned})
            continue
        contexts = (
            [{"previous_executed_semantic_action": item, "active_domain": "none"} for item in history]
            if partial["operation"] == "restore_previous"
            else [{"previous_executed_semantic_action": "none", "active_domain": item} for item in domains]
        )
        for context in contexts:
            result = bind(partial, context)
            again = bind(partial, context)
            if result != again:
                wrong.append({"id": case["id"], "nondeterministic": True, "context": context})
            expected_status = _expected_bind(partial, context)
            row = {
                "id": case["id"],
                "text": case["text"],
                "partial": partial,
                "context": context,
                "result": result,
                "expected_status": expected_status,
            }
            matrix.append(row)
            if result["status"] != expected_status or (
                expected_status == "bound" and result["command"] != _expected_command(partial, context)
            ):
                wrong.append(row)
            elif result["status"] == "bound":
                correct_bind += 1
            else:
                correct_unresolved += 1

    focused = {
        "Верни то, что было до этого": "restore_previous",
        "Вернись на одну назад": "relative_step",
    }
    focus_rows = [row for row in matrix if row["text"] in focused]
    return {
        "corpus": {
            "total": len(cases),
            "partial_positives": sum(case["role"] == "partial_positive" for case in cases),
            "full_controls": sum(case["role"] == "full_control" for case in cases),
            "adversarial_controls": sum(case["role"] == "adversarial_control" for case in cases),
        },
        "parser": {
            "precision": precision,
            "recall": recall,
            "true_partial": tp,
            "false_partial": fp,
            "missed_partial": fn,
            "true_negative": tn,
            "false_partial_cases": false_partial,
            "adversarial_false_partial": len(adversarial_false),
            "missed_cases": missed,
        },
        "binder": {
            "correct_bind": correct_bind,
            "correct_unresolved": correct_unresolved,
            "wrong_bind": len(wrong),
            "wrong_cases": wrong,
        },
        "focus": focus_rows,
        "parsed": parsed,
    }


def _expected_bind(partial: dict, context: dict) -> str:
    if partial["operation"] == "restore_previous":
        previous = context["previous_executed_semantic_action"]
        return "bound" if previous in _RESTORE_ALLOW else "unresolved"
    if partial["count"] != 1:
        return "unresolved"
    if context["active_domain"] not in _ORDERED_DOMAINS:
        return "unresolved"
    return "bound"


def _expected_command(partial: dict, context: dict) -> dict:
    if partial["operation"] == "restore_previous":
        return {"intent": _RESTORE_ALLOW[context["previous_executed_semantic_action"]]}
    intent = "media.previous" if partial["direction"] == "previous" else "media.next"
    return {"intent": intent}


def main() -> None:
    report = evaluate()
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    parser = report["parser"]
    binder = report["binder"]
    print(
        f"parser precision={parser['precision']} recall={parser['recall']} "
        f"false_partial={parser['false_partial']} missed={parser['missed_partial']}",
        flush=True,
    )
    print(
        f"binder correct_bind={binder['correct_bind']} "
        f"correct_unresolved={binder['correct_unresolved']} wrong_bind={binder['wrong_bind']}",
        flush=True,
    )
    if parser["false_partial_cases"] or parser["missed_cases"] or binder["wrong_cases"]:
        print(json.dumps({
            "false": parser["false_partial_cases"],
            "missed": parser["missed_cases"],
            "wrong": binder["wrong_cases"],
        }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
