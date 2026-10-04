"""Offline context-boundary probe. Does not use or change the production pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

from ollama_runner.nlu.backend import SEMANTIC_SCHEMA
from ollama_runner.nlu.normalize import semantic_outcome_from_dict
from ollama_runner.settings import get_settings


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "src" / "ollama_runner" / "prompts" / "semantic.md"
OUT = ROOT / "tests" / "context_boundary_v1.json"
MODELS = ("ministral-3:3b", "gemma3:4b")

CASES = {
    "B-55": "Замри на этом кадре",
    "B-70": "Пропусти этот кусок",
    "B-71": "Верни то, что было до этого",
    "B-73": "Вернись на одну назад",
}

# Only fields required by at least one of the four cases.
# active_domain: B-73, B-70. active_media_kind: B-55, B-70.
# segment_available: B-70. previous_semantic_action: B-71.
CONTEXTS = {
    "B-55": {
        "A_active_video": {
            "active_domain": "media.video",
            "active_media_kind": "video",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
        "B_active_audio": {
            "active_domain": "media.audio",
            "active_media_kind": "audio",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
        "C_no_media": {
            "active_domain": "none",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
    },
    "B-70": {
        "A_video_segment": {
            "active_domain": "media.video",
            "active_media_kind": "video",
            "segment_available": "true",
            "previous_semantic_action": "none",
        },
        "B_playlist_track": {
            "active_domain": "media.playlist",
            "active_media_kind": "audio",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
        "C_audio_no_segment": {
            "active_domain": "media.audio",
            "active_media_kind": "audio",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
        "D_no_media": {
            "active_domain": "none",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
    },
    "B-71": {
        "A_prev_media_next": {
            "active_domain": "none",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "media.next",
        },
        "B_prev_brightness_set": {
            "active_domain": "none",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "brightness.set",
        },
        "C_prev_turn_off": {
            "active_domain": "none",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "device.turn_off",
        },
        "D_no_previous": {
            "active_domain": "none",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
    },
    "B-73": {
        "A_playlist": {
            "active_domain": "media.playlist",
            "active_media_kind": "audio",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
        "B_episodic_video": {
            "active_domain": "media.episode",
            "active_media_kind": "video",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
        "C_photo_viewer": {
            "active_domain": "media.photos",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
        "D_no_domain": {
            "active_domain": "none",
            "active_media_kind": "none",
            "segment_available": "false",
            "previous_semantic_action": "none",
        },
    },
}


def _user_message(text: str, context: dict[str, str] | None) -> str:
    if context is None:
        return text
    lines = ["Context:"]
    for key, value in context.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("User:")
    lines.append(text)
    return "\n".join(lines)


def _view(outcome) -> dict:
    kind = outcome.kind
    command = getattr(outcome, "command", None)
    if command is None:
        return {
            "outcome": kind,
            "intent": None,
            "device_type": None,
            "mention": None,
            "area": None,
            "content": None,
            "value": None,
        }
    target = command.target
    return {
        "outcome": kind,
        "intent": command.intent,
        "device_type": target.device_type,
        "mention": target.mention,
        "area": target.area,
        "content": command.arguments.get("content"),
        "value": command.arguments.get("value"),
    }


def main() -> None:
    prompt = PROMPT.read_text(encoding="utf-8")
    if "## Различения" in prompt:
        raise SystemExit("prompt is not the frozen text-only prompt")
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    settings = get_settings()
    rows = []
    with httpx.Client(base_url=settings.ollama_url, timeout=120.0) as client:
        for model in MODELS:
            for case_id, text in CASES.items():
                variants = [("text_only", None), *CONTEXTS[case_id].items()]
                for name, context in variants:
                    user = _user_message(text, context)
                    response = client.post(
                        "/api/chat",
                        json={
                            "model": model,
                            "stream": False,
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": user},
                            ],
                            "format": SEMANTIC_SCHEMA,
                            "think": False,
                            "keep_alive": "30m",
                            "options": {"temperature": 0},
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    raw = body["message"]["content"]
                    try:
                        parsed = semantic_outcome_from_dict(json.loads(raw))
                        error = None
                        view = _view(parsed)
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        view = {"outcome": None, "intent": None}
                    row = {
                        "model": model,
                        "id": case_id,
                        "text": text,
                        "variant": name,
                        "context": context,
                        "parse_error": error,
                        **view,
                    }
                    rows.append(row)
                    print(
                        f"{model} {case_id} {name} {row.get('outcome')} {row.get('intent')}",
                        flush=True,
                    )
    OUT.write_text(
        json.dumps({"prompt_sha256": digest, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
