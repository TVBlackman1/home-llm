from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ollama_runner.main import run_home_request
from ollama_runner.request import EXECUTION_FAILED
from ollama_runner.settings import get_settings
from ollama_runner.ha.client import HaClient


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _extract_bearer(value: str | None) -> str | None:
    if value is None:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


@dataclass(frozen=True)
class _ApiContext:
    client: HaClient
    token: str
    timeout: float


def _speech_from_payload(payload: dict | None, default_error: str) -> str:
    payload = payload or {}
    status = str(payload.get("status") or "")
    reason = str(payload.get("reason") or "")
    outcome = str(payload.get("outcome") or "")

    if status == "clarify":
        if reason == "no_active_device":
            return "Уточните устройство: не нашел активное воспроизведение"
        return "Нужно уточнение команды"
    if status == "ambiguous":
        if reason == "media_type":
            return "Уточните тип медиа: аудио или видео"
        if reason == "device_type":
            return "Уточните тип устройства"
        return "Нашел несколько подходящих устройств, уточните команду"
    if status == "not_found":
        if reason == "unknown_owner":
            return "Не удалось определить владельца устройства"
        return "Не нашел подходящее устройство"
    if status == "unsupported":
        return "Команда распознана, но пока не поддерживается"
    if status == EXECUTION_FAILED:
        return "Не удалось выполнить команду в Home Assistant"
    if outcome == "not_command":
        return "Это не похоже на команду умного дома"
    if outcome == "needs_context":
        return "Недостаточно контекста для выполнения команды"
    return default_error


def _run_with_timeout(context: _ApiContext, text: str):
    return asyncio.run(
        asyncio.wait_for(
            asyncio.to_thread(run_home_request, context.client, text),
            timeout=context.timeout,
        )
    )


class _AssistHandler(BaseHTTPRequestHandler):
    server_version = "ollama-runner-assist/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Intentionally quiet: avoid leaking request data or auth headers.
        return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/assist":
            _json_response(
                self,
                HTTPStatus.NOT_FOUND,
                {
                    "success": False,
                    "speech": "Маршрут не найден",
                    "conversation_id": None,
                },
            )
            return

        context: _ApiContext = self.server.context  # type: ignore[attr-defined]
        token = _extract_bearer(self.headers.get("Authorization"))
        if token is None or token != context.token:
            _json_response(
                self,
                HTTPStatus.UNAUTHORIZED,
                {
                    "success": False,
                    "speech": "Ошибка авторизации",
                    "conversation_id": None,
                },
            )
            return

        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length)

        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            _json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {
                    "success": False,
                    "speech": "Некорректный JSON",
                    "conversation_id": None,
                },
            )
            return

        text = str(body.get("text") or "").strip()
        conversation_id = body.get("conversation_id")
        if not text:
            _json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {
                    "success": False,
                    "speech": "Пустая команда",
                    "conversation_id": conversation_id,
                },
            )
            return

        try:
            result = _run_with_timeout(context, text)
        except TimeoutError:
            _json_response(
                self,
                HTTPStatus.GATEWAY_TIMEOUT,
                {
                    "success": False,
                    "speech": "Превышено время ожидания ответа runtime",
                    "conversation_id": conversation_id,
                },
            )
            return
        except Exception:
            _json_response(
                self,
                HTTPStatus.BAD_GATEWAY,
                {
                    "success": False,
                    "speech": "Ошибка связи с semantic runtime",
                    "conversation_id": conversation_id,
                },
            )
            return

        if result.ok:
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "success": True,
                    "speech": "Готово",
                    "conversation_id": conversation_id,
                },
            )
            return

        speech = _speech_from_payload(result.payload, "Не удалось выполнить команду")
        _json_response(
            self,
            HTTPStatus.OK,
            {
                "success": False,
                "speech": speech,
                "conversation_id": conversation_id,
            },
        )


def main() -> None:
    settings = get_settings()
    if not settings.assist_api_token:
        raise SystemExit("ASSIST_API_TOKEN is empty")

    client = HaClient(settings)
    context = _ApiContext(
        client=client,
        token=settings.assist_api_token,
        timeout=settings.assist_api_timeout,
    )

    server = ThreadingHTTPServer((settings.assist_api_host, settings.assist_api_port), _AssistHandler)
    server.context = context  # type: ignore[attr-defined]

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        client.close()
