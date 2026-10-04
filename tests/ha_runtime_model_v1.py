"""Experiment-only generic HA runtime contract.

Not imported by production. Runtime context never invents a semantic
operation: navigation binding requires an already parsed relative_step,
and restore binding requires an already parsed restore_previous plus a record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ORDERED_CONTENT_TYPES = frozenset({"music", "playlist", "episode", "tvshow", "podcast"})

# Runtime feature required by an already chosen direction.
# Static RegistryDevice.capabilities are a different layer and are not consulted.
DIRECTION_FEATURE = {
    "previous": "PREVIOUS_TRACK",
    "forward": "NEXT_TRACK",
}
DIRECTION_INTENT = {
    "previous": "media.previous",
    "forward": "media.next",
}

REVERSIBLE_PROPERTIES = frozenset({"power", "brightness", "color", "volume"})
NOT_GENERICALLY_REVERSIBLE = frozenset({
    "media.next",
    "media.previous",
    "media.seek_forward",
    "media.seek_backward",
})


@dataclass(frozen=True)
class MediaRuntimeSnapshot:
    """Standard media_player state. None means unknown, never implied support."""

    state: str | None
    supported_features: frozenset[str] | None
    media_content_type: str | None
    media_content_id: str | None
    media_position: float | None
    media_duration: float | None
    volume_level: float | None


@dataclass(frozen=True)
class RelativeStep:
    direction: str
    count: int


@dataclass(frozen=True)
class NavigationResult:
    kind: str
    intent: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class UndoRecord:
    semantic_action: str
    target: str
    property: str
    before: object
    after: object
    executed_at: str


@dataclass(frozen=True)
class RestoreCommand:
    property: str
    value: object
    via: str = "recorded_before"


@dataclass(frozen=True)
class RestoreResult:
    kind: str
    command: RestoreCommand | None = None
    reason: str | None = None


def snapshot_from_raw(raw: dict) -> MediaRuntimeSnapshot:
    """Load a fixture. Static capability lists are dropped on purpose."""
    features = raw.get("supported_features", None)
    if features is not None:
        features = frozenset(features)
    position = raw.get("media_position", None)
    duration = raw.get("media_duration", None)
    volume = raw.get("volume_level", None)
    return MediaRuntimeSnapshot(
        state=raw.get("state", None),
        supported_features=features,
        media_content_type=raw.get("media_content_type", None),
        media_content_id=raw.get("media_content_id", None),
        media_position=None if position is None else float(position),
        media_duration=None if duration is None else float(duration),
        volume_level=None if volume is None else float(volume),
    )


def session_facts(snapshot: MediaRuntimeSnapshot, feature: str) -> dict[str, bool]:
    """Observable facts only. This does not name a semantic operation."""
    features_known = snapshot.supported_features is not None
    return {
        "state_known": snapshot.state is not None,
        "playing": snapshot.state == "playing",
        "features_known": features_known,
        "feature_present": features_known and feature in snapshot.supported_features,
        "ordered_content": snapshot.media_content_type in ORDERED_CONTENT_TYPES,
        "content_type_known": snapshot.media_content_type is not None,
    }


def bind_navigation(step: RelativeStep, sessions: tuple[MediaRuntimeSnapshot, ...]) -> NavigationResult:
    if step.count != 1:
        return NavigationResult(kind="REPRESENTATION_LIMIT")
    feature = DIRECTION_FEATURE[step.direction]
    active: list[MediaRuntimeSnapshot] = []
    for session in sessions:
        facts = session_facts(session, feature)
        if facts["playing"] and facts["feature_present"] and facts["ordered_content"]:
            active.append(session)
    if len(active) == 1:
        return NavigationResult(kind="BOUND", intent=DIRECTION_INTENT[step.direction])
    if len(active) > 1:
        return NavigationResult(kind="UNRESOLVED", reason="multiple_active_sessions")
    return NavigationResult(kind="UNRESOLVED", reason=_negative_reason(sessions, feature))


def _negative_reason(sessions: tuple[MediaRuntimeSnapshot, ...], feature: str) -> str:
    if not sessions:
        return "no_active_session"
    playing_known_feature = []
    playing_missing_feature = []
    playing_unknown_feature = []
    unknown_state = []
    incompatible_state = []
    for session in sessions:
        facts = session_facts(session, feature)
        if facts["playing"] and facts["feature_present"] and not facts["ordered_content"]:
            playing_known_feature.append(session)
        elif facts["playing"] and facts["features_known"] and not facts["feature_present"]:
            playing_missing_feature.append(session)
        elif facts["playing"] and not facts["features_known"]:
            playing_unknown_feature.append(session)
        elif not facts["state_known"]:
            unknown_state.append(session)
        elif not facts["playing"]:
            incompatible_state.append(session)
    if playing_known_feature:
        return "navigation_semantics_unknown"
    if playing_missing_feature:
        return "capability_missing"
    if playing_unknown_feature:
        return "capability_unknown"
    if unknown_state:
        return "state_unknown"
    if incompatible_state:
        return "state_incompatible"
    return "no_active_session"


def _read_property(property_name: str, current: dict) -> object | None:
    if property_name == "brightness":
        return current.get("brightness") if "brightness" in current else None
    if property_name == "volume":
        return current.get("volume_level") if "volume_level" in current else None
    if property_name == "power":
        return current.get("state") if "state" in current else None
    if property_name == "color":
        if "color_mode" not in current or "rgb_color" not in current:
            return None
        return {"color_mode": current["color_mode"], "rgb_color": current["rgb_color"]}
    return None


def bind_restore(record: UndoRecord | None, current: dict) -> RestoreResult:
    """Restore recorded before-state. There is no inverse(intent) path."""
    if record is None:
        return RestoreResult(kind="UNRESOLVED", reason="no_record")
    if record.semantic_action in NOT_GENERICALLY_REVERSIBLE:
        return RestoreResult(kind="UNRESOLVED", reason="not_reversible")
    if record.property not in REVERSIBLE_PROPERTIES:
        return RestoreResult(kind="UNRESOLVED", reason="not_reversible")
    if record.before is None or record.after is None:
        return RestoreResult(kind="UNRESOLVED", reason="not_reversible")
    observed = _read_property(record.property, current)
    if observed is None:
        return RestoreResult(kind="UNRESOLVED", reason="state_unknown")
    if observed != record.after:
        return RestoreResult(kind="UNRESOLVED", reason="conflict")
    return RestoreResult(
        kind="RESTORE",
        command=RestoreCommand(property=record.property, value=record.before),
    )


def _matches(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def evaluate(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    navigation_rows = []
    wrong = 0
    for case in document["cases"]:
        sessions = tuple(snapshot_from_raw(raw) for raw in case["sessions"])
        for index, check in enumerate(case["checks"], start=1):
            result = bind_navigation(
                RelativeStep(direction=check["direction"], count=check["count"]),
                sessions,
            )
            actual = {"kind": result.kind, "intent": result.intent, "reason": result.reason}
            expected = check["expected"]
            ok = _matches(actual, expected) and "entity_id" not in actual
            if not ok:
                wrong += 1
            navigation_rows.append({
                "id": case["id"],
                "check": index,
                "direction": check["direction"],
                "count": check["count"],
                "expected": expected,
                "actual": {key: value for key, value in actual.items() if value is not None},
                "ok": ok,
            })

    restore_rows = []
    for case in document["restore_cases"]:
        raw = case["record"]
        record = None if raw is None else UndoRecord(**raw)
        result = bind_restore(record, case["current"])
        if result.command is None:
            actual = {"kind": result.kind, "reason": result.reason}
        else:
            actual = {
                "kind": result.kind,
                "property": result.command.property,
                "value": result.command.value,
                "via": result.command.via,
            }
        ok = _matches(actual, case["expected"])
        if not ok:
            wrong += 1
        restore_rows.append({
            "id": case["id"],
            "expected": case["expected"],
            "actual": actual,
            "ok": ok,
        })

    return {
        "version": document["version"],
        "navigation_checks": len(navigation_rows),
        "navigation_ok": sum(1 for row in navigation_rows if row["ok"]),
        "restore_checks": len(restore_rows),
        "restore_ok": sum(1 for row in restore_rows if row["ok"]),
        "wrong": wrong,
        "navigation": navigation_rows,
        "restore": restore_rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    summary = evaluate(root / "ha_runtime_fixtures_v1.json")
    out = root / "ha_runtime_model_v1_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"navigation {summary['navigation_ok']}/{summary['navigation_checks']} "
        f"restore {summary['restore_ok']}/{summary['restore_checks']} "
        f"wrong {summary['wrong']}"
    )
    if summary["wrong"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
