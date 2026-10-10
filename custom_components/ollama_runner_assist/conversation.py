from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

from .const import CONF_API_TOKEN, CONF_RUNTIME_URL, CONF_TIMEOUT, DEFAULT_TIMEOUT, DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities,
) -> None:
    _LOGGER.debug("[%s] conversation.async_setup_entry entry_id=%s", DOMAIN, entry.entry_id)
    async_add_entities([OllamaRunnerAssistConversationEntity(entry)])
    _LOGGER.debug("[%s] ConversationEntity scheduled for add entry_id=%s", DOMAIN, entry.entry_id)


class OllamaRunnerAssistConversationEntity(ConversationEntity):
    _attr_has_entity_name = True
    _attr_name = "Ollama Runner Assist"

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        self._session: aiohttp.ClientSession | None = None
        self._runtime_url = str(entry.data[CONF_RUNTIME_URL]).rstrip("/")
        self._token = str(entry.data[CONF_API_TOKEN])
        self._timeout = float(entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        _LOGGER.debug("[%s] ConversationEntity init entry_id=%s", DOMAIN, entry.entry_id)

    @property
    def supported_languages(self) -> list[str] | str:
        """Supported languages for this conversation agent.

        Home Assistant ConversationEntity requires this property.
        """

        return ["ru", "ru-RU"]

    async def async_added_to_hass(self) -> None:
        _LOGGER.debug("[%s] ConversationEntity async_added_to_hass", DOMAIN)
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        )

    async def async_will_remove_from_hass(self) -> None:
        _LOGGER.debug("[%s] ConversationEntity async_will_remove_from_hass", DOMAIN)
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        del chat_log
        response = intent.IntentResponse(language=user_input.language)

        payload = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id,
        }
        body: dict[str, Any]

        try:
            body = await self._call_runtime(payload)
        except asyncio.TimeoutError:
            _LOGGER.warning("[%s] runtime timeout", DOMAIN)
            response.async_set_speech("Сервер semantic runtime не ответил вовремя")
            return conversation.ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )
        except aiohttp.ClientError:
            _LOGGER.warning("[%s] runtime connection error", DOMAIN)
            response.async_set_speech("Не удалось связаться с semantic runtime")
            return conversation.ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )
        except Exception:
            _LOGGER.exception("[%s] runtime call failed", DOMAIN)
            response.async_set_speech("Внутренняя ошибка интеграции semantic runtime")
            return conversation.ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )

        speech = str(body.get("speech") or "Не удалось выполнить команду")
        success = bool(body.get("success"))
        conversation_id = body.get("conversation_id") or user_input.conversation_id

        response.async_set_speech(speech)
        return conversation.ConversationResult(
            response=response,
            conversation_id=conversation_id,
            continue_conversation=not success,
        )

    async def _call_runtime(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )

        url = f"{self._runtime_url}/api/assist"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        assert self._session is not None
        async with self._session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                _LOGGER.warning("[%s] runtime non-200 status=%s", DOMAIN, resp.status)
                return {
                    "success": False,
                    "speech": "Сервер semantic runtime вернул ошибку",
                    "conversation_id": payload.get("conversation_id"),
                }
            try:
                data = await resp.json(content_type=None)
            except Exception:
                _LOGGER.exception("[%s] runtime invalid json", DOMAIN)
                return {
                    "success": False,
                    "speech": "Некорректный ответ semantic runtime",
                    "conversation_id": payload.get("conversation_id"),
                }
            if not isinstance(data, dict):
                return {
                    "success": False,
                    "speech": "Некорректный ответ semantic runtime",
                    "conversation_id": payload.get("conversation_id"),
                }
            return data
