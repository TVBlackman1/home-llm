from __future__ import annotations

import base64
import json
import os
import socket
import struct
from urllib.parse import urlparse

import httpx

from ollama_runner.settings import Settings


class HaClient:
    """REST for states and service calls, websocket for registries.

    The split matches Home Assistant: states and services are HTTP, areas and
    devices are registry commands.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.ha_token:
            raise RuntimeError("HA_TOKEN is empty")
        self._url = settings.ha_url.rstrip("/")
        self._token = settings.ha_token
        self._http = httpx.Client(
            base_url=self._url,
            timeout=settings.ollama_timeout,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._http.close()

    def get_states(self) -> list[dict]:
        return self._get("/api/states")

    def get_services(self) -> list[dict]:
        return self._get("/api/services")

    def get_areas(self) -> list[dict]:
        return self._registry("config/area_registry/list")

    def get_devices(self) -> list[dict]:
        return self._registry("config/device_registry/list")

    def get_entities(self) -> list[dict]:
        return self._registry("config/entity_registry/list")

    def get_entity(self, entity_id: str) -> dict:
        return self._registry(
            "config/entity_registry/get",
            entity_id=entity_id,
        )

    def get_labels(self) -> list[dict]:
        return self._registry("config/label_registry/list")

    def call_service(self, domain: str, service: str, data: dict) -> None:
        response = self._http.post(f"/api/services/{domain}/{service}", json=data)
        response.raise_for_status()

    def _get(self, path: str) -> list[dict]:
        response = self._http.get(path)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"{path} did not return a list")
        return payload

    def _registry(self, command: str, **fields) -> dict | list:
        parsed = urlparse(self._url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        results = _websocket_call(
            host,
            port,
            secure=parsed.scheme == "https",
            token=self._token,
            commands=[{"id": 1, "type": command, **fields}],
        )
        message = results[1]
        if not message.get("success"):
            raise RuntimeError(f"{command} failed: {message.get('error')}")
        return message.get("result")


def _websocket_call(
    host: str,
    port: int,
    *,
    secure: bool,
    token: str,
    commands: list[dict],
) -> dict[int, dict]:
    sock = socket.create_connection((host, port), timeout=30)
    if secure:
        import ssl

        sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        "GET /api/websocket HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(request.encode())
    header = b""
    while b"\r\n\r\n" not in header:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Home Assistant closed the websocket handshake")
        header += chunk
    status = header.split(b"\r\n", 1)[0]
    if b"101" not in status:
        raise ConnectionError(status.decode(errors="replace"))
    pending = bytearray(header.split(b"\r\n\r\n", 1)[1])

    def read_message() -> dict:
        nonlocal pending
        while True:
            frame = _read_frame(sock, pending)
            if frame is None:
                continue
            return json.loads(frame)

    hello = read_message()
    if hello.get("type") != "auth_required":
        raise ConnectionError(f"unexpected websocket hello: {hello.get('type')}")
    _send_frame(sock, json.dumps({"type": "auth", "access_token": token}))
    auth = read_message()
    if auth.get("type") != "auth_ok":
        raise ConnectionError(auth.get("message") or "Home Assistant rejected the token")

    waiting = {command["id"] for command in commands}
    for command in commands:
        _send_frame(sock, json.dumps(command))
    found: dict[int, dict] = {}
    while waiting:
        message = read_message()
        message_id = message.get("id")
        if message_id in waiting:
            found[message_id] = message
            waiting.remove(message_id)
    sock.close()
    return found


def _send_frame(sock: socket.socket, text: str) -> None:
    payload = text.encode()
    mask = os.urandom(4)
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)


def _read_frame(sock: socket.socket, pending: bytearray) -> str | None:
    while len(pending) < 2:
        pending.extend(_recv(sock))
    length = pending[1] & 0x7F
    header_len = 2
    if length == 126:
        while len(pending) < 4:
            pending.extend(_recv(sock))
        length = struct.unpack("!H", pending[2:4])[0]
        header_len = 4
    elif length == 127:
        while len(pending) < 10:
            pending.extend(_recv(sock))
        length = struct.unpack("!Q", pending[2:10])[0]
        header_len = 10
    if pending[1] & 0x80:
        header_len += 4
    while len(pending) < header_len + length:
        pending.extend(_recv(sock))
    frame = bytes(pending[: header_len + length])
    del pending[: header_len + length]
    opcode = frame[0] & 0x0F
    if opcode == 0x1:
        return frame[header_len:].decode()
    if opcode == 0x8:
        raise ConnectionError("Home Assistant closed the websocket")
    if opcode == 0x9:
        _send_frame(sock, "")
    return None


def _recv(sock: socket.socket) -> bytes:
    chunk = sock.recv(65536)
    if not chunk:
        raise ConnectionError("Home Assistant closed the websocket")
    return chunk
