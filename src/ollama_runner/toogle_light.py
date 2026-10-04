from __future__ import annotations

from ollama_runner.settings import get_settings


ENTITY_ID = "light.lampa"
DELAY = 1.0


def call_service(domain: str, service: str, **data) -> None:
    settings = get_settings()
    if not settings.ha_token:
        raise RuntimeError("HA_TOKEN is empty")
    import requests

    response = requests.post(
        f"{settings.ha_url}/api/services/{domain}/{service}",
        json=data,
        headers={
            "Authorization": f"Bearer {settings.ha_token}",
            "Content-Type": "application/json",
        },
        timeout=5,
    )
    response.raise_for_status()


def toggle_light() -> None:
    call_service("light", "toggle", entity_id=ENTITY_ID)


def animation() -> None:
    import time

    toggle_light()
    time.sleep(DELAY)
    toggle_light()


def main() -> None:
    animation()


if __name__ == "__main__":
    main()
