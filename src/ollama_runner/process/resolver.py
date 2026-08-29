from __future__ import annotations

from ollama_runner.embeeding import resolve as resolve_scores
from ollama_runner.protocols import Inventory
from ollama_runner.types import Command, Intent


class FastTextResolver:
    def __init__(self, model, inventory: Inventory) -> None:
        self._model = model
        self._inventory = inventory

    def resolve(self, intent: Intent) -> Command:
        results, _, _ = resolve_scores(
            self._model,
            owner_text=intent.owner.strip() or "общий",
            device_text=intent.device.strip(),
            place_text=intent.place.strip(),
            devices=self._inventory.devices(),
            owners=self._inventory.owners(),
        )
        chosen = results[0][0]

        return Command(
            device_id=chosen.id,
            action=intent.action,
            value=intent.value,
            owner=intent.owner,
            place=intent.place,
        )
