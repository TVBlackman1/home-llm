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

## Интеграция с Home Assistant Assist

Ниже минимальная схема: Assist -> custom ConversationEntity -> HTTP API runtime -> `run_home_request`.

### 1) Запуск HTTP API на сервере semantic runtime

Добавь в `.env` (или переменные окружения):

```env
ASSIST_API_HOST=0.0.0.0
ASSIST_API_PORT=8765
ASSIST_API_TOKEN=ваш_секретный_токен
ASSIST_API_TIMEOUT=20
```

Запуск:

```bash
uv run assist-api
```

Проверка:

```bash
curl -X POST "http://<RUNTIME_HOST>:8765/api/assist" \
  -H "Authorization: Bearer <ASSIST_API_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"text":"включи свет", "conversation_id": null}'
```

### 2) Установка custom integration в Home Assistant

Скопируй директорию:

`custom_components/ollama_runner_assist/`

в каталог Home Assistant:

`/config/custom_components/ollama_runner_assist/`

Перезапусти Home Assistant.

### 3) Настройка интеграции

В HA: **Settings -> Devices & Services -> Add Integration** -> `Ollama Runner Assist`.

Поля:
- `runtime_url`: `http://<RUNTIME_HOST>:8765`
- `api_token`: тот же `ASSIST_API_TOKEN`
- `timeout`: например `20`

### 4) Выбор conversation agent в Assist

В настройках Assist выбери агент `Ollama Runner Assist`.

### 5) Проверка выполнения

Скажи через Assist: «включи свет в спальне».  
Ожидаемо:
- команда уходит в runtime;
- выполнение происходит через существующий `HaExecutor`;
- Assist получает речевой ответ (`Готово` только при успешном выполнении).

### Примечание по сети и безопасности

- API рассчитан на доступ из локальной сети (HAOS -> runtime).
- Не публикуй порт API в интернет.
- Ограничь доступ firewall/VPN/сегментацией сети.
