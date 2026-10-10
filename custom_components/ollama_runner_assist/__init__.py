from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CONVERSATION]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    _LOGGER.debug("[%s] async_setup", DOMAIN)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    _LOGGER.debug("[%s] async_setup_entry entry_id=%s", DOMAIN, entry.entry_id)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = dict(entry.data)
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception:
        _LOGGER.exception(
            "[%s] failed to forward entry setup for conversation platform entry_id=%s",
            DOMAIN,
            entry.entry_id,
        )
        return False
    _LOGGER.debug("[%s] conversation platform forwarded entry_id=%s", DOMAIN, entry.entry_id)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    _LOGGER.debug("[%s] async_unload_entry entry_id=%s", DOMAIN, entry.entry_id)
    ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        _LOGGER.debug("[%s] unloaded entry_id=%s", DOMAIN, entry.entry_id)
    return ok
