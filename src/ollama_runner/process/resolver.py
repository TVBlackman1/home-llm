from __future__ import annotations

from ollama_runner.embeeding import (
    canonical_owner,
    canonical_place,
    resolve as resolve_scores,
)
from ollama_runner.protocols import Inventory
from ollama_runner.types import Command, Intent


class FastTextResolver:
    def __init__(self, model, inventory: Inventory) -> None:
        self._model = model
        self._inventory = inventory

    def resolve(self, intent: Intent) -> Command:
        devices = self._inventory.devices()
        owners = self._inventory.owners()
        results, _, _ = resolve_scores(
            self._model,
            owner_text=intent.owner.strip() or "общий",
            device_text=intent.device.strip(),
            place_text=intent.place.strip(),
            devices=devices,
            owners=owners,
        )
        chosen = results[0][0]

        return Command(
            device_id=chosen.id,
            action=intent.action,
            value=intent.value,
            owner=canonical_owner(self._model, intent.owner, owners=owners),
            place=canonical_place(self._model, intent.place, devices=devices),
        )
