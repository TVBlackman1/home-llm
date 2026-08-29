# ollama-runner

Пайплайн голосовых команд умного дома: текст или Whisper → кэш Postgres → LLM + FastText → конкретный `device_id` → HA (пока заглушка) или pytest.

## Окружение

Скопируй `.env.example` в `.env` и поправь значения.

| Переменная | Назначение |
|---|---|
| `OLLAMA_HOST` / `OLLAMA_PORT` | Ollama |
| `OLLAMA_MODEL` | Модель по умолчанию |
| `OLLAMA_TEMPERATURE` / `OLLAMA_KEEP_ALIVE` / `OLLAMA_TIMEOUT` | Inference |
| `PROMPT_PATH` | Системный промпт |
| `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Кэш команд |
| `WHISPER_HOST` / `WHISPER_PORT` / `WHISPER_PATH` / `WHISPER_TIMEOUT` | ASR-сервер |
| `FASTTEXT_MODEL_PATH` | Векторы, по умолчанию `cc.ru.300.bin` |
| `HA_HOST` / `HA_PORT` / `HA_TOKEN` | Home Assistant (сток-заглушка) |

Для продукта: Ollama, Postgres, FastText-файл. Whisper — только для `--audio`.
Тесты LLM + embedding Postgres не используют.

## Запуск

```bash
uv sync
uv run ollama-runner --text "Включи свет Маше"
uv run ollama-runner --file cmd.txt
uv run ollama-runner --audio cmd.wav --sink ha
uv run ollama-runner --model ministral-3:3b   # интерактив, модель из .env если не указана
```

Разовый резолвер без LLM:

```bash
uv run emb --owner Маша --device свет --place спальня
```

## Тесты

Общий формат кейса: `id`, `text`, `intent`, `expected`.

- `tests/embed_cases.json` — только vector embedding (свой инвентарь, крайние случаи, с локацией и без)
- `tests/cases.json` — команды для LLM + embedding
- pipeline-прогон берёт оба набора (дубли по `text` схлопываются)

```bash
# билдер / кэш в памяти, без серверов
uv run pytest tests/test_pipeline.py -m unit -q

# embedding: нужен cc.ru.300.bin
uv run pytest tests/test_embed.py -m embed -q

# LLM + embedding: Ollama + FastText, без Postgres
# каждый кейс — 5 независимых запросов без накопления контекста
uv run pytest tests/test_mistral.py -m pipeline -q

# остальные модели
uv run pytest tests/test_qwen35.py tests/test_gemma3.py tests/test_qwen25.py tests/test_phi4mini.py
```

Пересобрать embed-кейсы после изменения инвентаря:

```bash
uv run python tests/gen_cases.py
```
