import os
import time

import requests


HA_URL = "http://192.168.8.168"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0MTQ1Mjk4NmIxNmY0MGZlOGM0ZTBlZjc3MDM3OTI5MyIsImlhdCI6MTc4ODA0NjE5MSwiZXhwIjoyMTAzNDA2MTkxfQ.v7ue-ckZjNNWV6b1hrGpmmIuRP4a0fMhjYJjijS2Ghc"

ENTITY_ID = "light.lampa"
DELAY = 1.0


session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
})


def call_service(domain: str, service: str, **data) -> None:
    response = session.post(
        f"{HA_URL}/api/services/{domain}/{service}",
        json=data,
        timeout=5,
    )
    response.raise_for_status()


def toggle_light() -> None:
    call_service(
        "light",
        "toggle",
        entity_id=ENTITY_ID,
    )

def animation():
    toggle_light()
    time.sleep(DELAY)
    toggle_light()    

def main() -> None:
    animation()


if __name__ == "__main__":
    main()