"""Structured result and compact trace for one production request.

Statuses reuse the resolver and NLU vocabulary. `success` means a resolved
command was accepted by Home Assistant. `execution_failed` means resolution
succeeded and the service call did not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ollama_runner.semantic import SemanticCommand
from ollama_runner.types import Result


SUCCESS = "success"
EXECUTION_FAILED = "execution_failed"

_LOGGER = logging.getLogger("ollama_runner.request")
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_AUTHORIZATION = re.compile(r"(?i)\bauthorization\s*[:=]\s*\S+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class CommandResult:
    """One atomic command inside a request."""

    status: str
    intent: str = ""
    source: str = ""
    decision: str = ""
    device_type: str | None = None
    mention: str | None = None
    area: str | None = None
    ordinal: int | None = None
    resolution_status: str = ""
    device_id: str | None = None
    resolved_type: str | None = None
    resolved_area: str | None = None
    entity_id: str | None = None
    service: str | None = None
    reason: str = ""
    candidates: tuple[str, ...] = ()
    http_status: int | None = None
    executed: bool = False


@dataclass(frozen=True)
class RequestResult:
    """What a caller can keep after one utterance, including every command."""

    request_id: int
    text: str
    status: str
    commands: tuple[CommandResult, ...]
    parse_reason: str = ""
    fully_parsed: bool = False
    unresolved: tuple[str, ...] = ()
    llm: dict | None = None
    merge_notes: tuple[str, ...] | None = None

    @property
    def ok(self) -> bool:
        return self.status == SUCCESS


def overall_status(commands: tuple[CommandResult, ...] | list[CommandResult]) -> str:
    if not commands:
        return "not_command"
    for command in commands:
        if command.status != SUCCESS:
            return command.status
    return SUCCESS


def command_from_result(
    *,
    source: str,
    decision: str,
    result: Result,
    semantic: SemanticCommand | None = None,
    resolution_status: str = "",
    device_id: str | None = None,
    resolved_type: str | None = None,
    resolved_area: str | None = None,
    candidates: tuple[str, ...] = (),
    reason: str = "",
) -> CommandResult:
    """Map one existing Result onto a command outcome. Does not re-resolve."""

    payload = result.payload or {}
    execution = payload.get("execution") or {}
    payload_status = str(payload.get("status") or "")
    payload_reason = str(payload.get("reason") or reason or "")
    if payload_status == EXECUTION_FAILED:
        status = EXECUTION_FAILED
    elif execution.get("attempted") and result.ok and payload_status == "resolved":
        status = SUCCESS
    else:
        status = payload_status or decision
    target = semantic.target if semantic is not None else None
    http_status = execution.get("http_status")
    return CommandResult(
        status=status,
        intent=semantic.intent if semantic is not None else str(payload.get("intent") or ""),
        source=source,
        decision=decision,
        device_type=target.device_type if target is not None else None,
        mention=target.mention if target is not None else None,
        area=target.area if target is not None else None,
        ordinal=target.ordinal if target is not None else None,
        resolution_status=resolution_status,
        device_id=device_id,
        resolved_type=resolved_type,
        resolved_area=resolved_area,
        entity_id=execution.get("entity_id") or None,
        service=execution.get("service") or None,
        reason=payload_reason,
        candidates=candidates,
        http_status=http_status if isinstance(http_status, int) else None,
        executed=bool(execution.get("attempted")),
    )


def configure_request_logging(level: int = logging.INFO) -> None:
    """Show the production trace on the CLI. Does not attach to other loggers."""

    _LOGGER.setLevel(level)
    _LOGGER.propagate = False
    if _LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)


def redact(text: str) -> str:
    """Drop credentials before a string can reach a log line or a result reason."""

    cleaned = _BEARER.sub("bearer [redacted]", text)
    cleaned = _AUTHORIZATION.sub("authorization [redacted]", cleaned)
    return _JWT.sub("[redacted]", cleaned)


def emit_request(request: RequestResult) -> None:
    """INFO is the short trace. DEBUG adds parser, model, merge and candidates."""

    req = request.request_id
    _LOGGER.info("[request] req=%s %s", req, redact(request.text))
    _LOGGER.debug(
        "[parse] req=%s %s",
        req,
        _fields(
            fully_parsed=request.fully_parsed,
            reason=request.parse_reason,
            commands=len(request.commands),
            unresolved=",".join(request.unresolved),
        ),
    )
    if request.llm is not None:
        _LOGGER.debug("[llm] req=%s %s", req, _fields(**request.llm))
    if request.merge_notes is not None:
        _LOGGER.debug(
            "[merge] req=%s %s",
            req,
            _fields(notes="; ".join(request.merge_notes)),
        )
    several = len(request.commands) > 1
    for index, command in enumerate(request.commands, start=1):
        prefix = {"n": index} if several else {}
        _LOGGER.info("[semantic] req=%s %s", req, _semantic_line(command, prefix))
        if command.resolution_status:
            _LOGGER.info("[resolve] req=%s %s", req, _resolve_line(command, prefix))
            _LOGGER.debug(
                "[resolve] req=%s %s",
                req,
                _fields(
                    **prefix,
                    candidates=",".join(command.candidates),
                    reason=command.reason,
                ),
            )
        if command.executed:
            _LOGGER.info("[execute] req=%s %s", req, _execute_line(command, prefix))
            if command.http_status is not None:
                _LOGGER.debug(
                    "[execute] req=%s %s",
                    req,
                    _fields(**prefix, http_status=command.http_status),
                )
        _LOGGER.info("[result] req=%s %s", req, _result_line(command, prefix))


def _semantic_line(command: CommandResult, prefix: dict) -> str:
    fields = dict(prefix)
    fields["source"] = command.source
    if command.source == "llm" and command.decision:
        fields["decision"] = command.decision
    if command.intent:
        fields["intent"] = command.intent
    fields["device_type"] = command.device_type
    fields["mention"] = command.mention
    fields["area"] = command.area
    if command.ordinal is not None:
        fields["ordinal"] = command.ordinal
    return _fields(**fields)


def _resolve_line(command: CommandResult, prefix: dict) -> str:
    return _fields(
        **prefix,
        status=command.resolution_status,
        device=command.device_id,
        type=command.resolved_type,
        area=command.resolved_area,
        reason=command.reason if command.resolution_status != "resolved" else "",
    )


def _execute_line(command: CommandResult, prefix: dict) -> str:
    return _fields(
        **prefix,
        intent=command.intent,
        service=command.service,
        entity=command.entity_id,
    )


def _result_line(command: CommandResult, prefix: dict) -> str:
    reason = command.reason if command.status == EXECUTION_FAILED else ""
    return _fields(**prefix, status=command.status, reason=reason)


def _fields(**items) -> str:
    parts: list[str] = []
    for key, value in items.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={redact(str(value))}")
    return " ".join(parts)
