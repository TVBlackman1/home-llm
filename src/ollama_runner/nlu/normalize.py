from __future__ import annotations

from ollama_runner.inventory.registry import (
    canonicalize_device_type,
    ordinal_from_mention,
    type_from_mention,
)
from ollama_runner.semantic import (
    CommandOutcome,
    NeedsContextOutcome,
    NotCommandOutcome,
    SemanticCommand,
    SemanticNLUOutcome,
    Target,
)


_CONTEXT_INTENTS = frozenset({
    "media.next",
    "media.previous",
    "media.pause",
    "media.resume",
})
_CONTENT_INTENTS = frozenset({"audio.play", "video.play", "content.play", "photos.show"})


def _blank(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _discarded_intent(data: dict) -> str | None:
    raw = data.get("intent")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def semantic_outcome_from_dict(data: dict) -> SemanticNLUOutcome:
    """COMMAND carries a household intent. The other two outcomes cannot."""

    outcome = str(data.get("outcome", "")).strip()
    if outcome == "not_command":
        return NotCommandOutcome(discarded_intent=_discarded_intent(data))
    if outcome == "needs_context":
        return NeedsContextOutcome(discarded_intent=_discarded_intent(data))
    if outcome == "command":
        return CommandOutcome(semantic_command_from_dict(data))
    return NeedsContextOutcome(discarded_intent=_discarded_intent(data))


def semantic_command_from_dict(data: dict) -> SemanticCommand:
    intent = str(data["intent"]).strip()
    mention = _blank(data.get("mention"))
    mentioned_type = type_from_mention(mention)
    parsed_type = canonicalize_device_type(_blank(data.get("device_type")))
    device_type = mentioned_type or parsed_type

    ordinal_raw = data.get("ordinal")
    ordinal = int(ordinal_raw) if ordinal_raw else None
    if ordinal is None:
        ordinal = ordinal_from_mention(mention)

    owner = _blank(data.get("owner"))
    area = _blank(data.get("area"))

    # Pause/next without a spoken device must not keep a model-invented TV.
    if intent in _CONTEXT_INTENTS and not mention:
        device_type = None
        owner = None
        area = None
        ordinal = None

    explicit = any((device_type, mention, owner, area, ordinal is not None))

    arguments: dict[str, str] = {}
    content = _blank(data.get("content"))
    value = _blank(data.get("value"))
    if content and intent in _CONTENT_INTENTS:
        arguments["content"] = content
    if value:
        arguments["value"] = value

    return SemanticCommand(
        intent=intent,
        target=Target(
            device_type=device_type,
            mention=mention,
            owner=owner,
            area=area,
            ordinal=ordinal,
            explicit=explicit,
        ),
        arguments=arguments,
    )
