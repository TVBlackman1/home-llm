from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult

from .const import CONF_API_TOKEN, CONF_RUNTIME_URL, CONF_TIMEOUT, DEFAULT_TIMEOUT, DOMAIN

_LOGGER = logging.getLogger(__name__)


class OllamaRunnerAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None) -> FlowResult:
        if user_input is not None:
            _LOGGER.debug("[%s] config flow create entry", DOMAIN)
            return self.async_create_entry(
                title="Ollama Runner Assist",
                data=user_input,
            )

        schema = vol.Schema(
            {
                vol.Required(CONF_RUNTIME_URL): str,
                vol.Required(CONF_API_TOKEN): str,
                vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.Coerce(float),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema)
