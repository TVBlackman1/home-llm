from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from psycopg.conninfo import make_conninfo
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    ollama_model: str = "ministral-3:3b"
    ollama_temperature: float = 0.0
    ollama_keep_alive: str = "30m"
    ollama_timeout: float = 300.0
    prompt_path: str = "src/ollama_runner/prompts/homeassist.md"

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str = "ollama"
    postgres_password: str = "ollama"
    postgres_db: str = "ollama_runner"

    whisper_host: str = "127.0.0.1"
    whisper_port: int = 9000
    whisper_path: str = "/v1/audio/transcriptions"
    whisper_timeout: float = 120.0

    fasttext_model_path: str = "cc.ru.300.bin"

    ha_host: str = "127.0.0.1"
    ha_port: int = 8123
    ha_token: str = ""

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def whisper_url(self) -> str:
        return f"http://{self.whisper_host}:{self.whisper_port}"

    @property
    def ha_url(self) -> str:
        return f"http://{self.ha_host}:{self.ha_port}"

    @property
    def resolved_prompt_path(self) -> Path:
        path = Path(self.prompt_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def resolved_fasttext_path(self) -> Path:
        path = Path(self.fasttext_model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def postgres_conninfo(self) -> str:
        return make_conninfo(
            host=self.postgres_host,
            port=self.postgres_port,
            user=self.postgres_user,
            password=self.postgres_password,
            dbname=self.postgres_db,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
