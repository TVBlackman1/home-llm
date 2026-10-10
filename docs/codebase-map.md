# Карта кодовой базы (быстрый вход)

Цель файла: помочь выбрать **минимальный** набор файлов под конкретную задачу.

## 1) Если нужно понять, где стартует выполнение

- CLI-вход: `src/ollama_runner/main.py`
- Сборка пайплайнов и shared-ресурсов: `src/ollama_runner/factory.py`
- Конфиг из `.env`: `src/ollama_runner/settings.py`

Обычно этого достаточно для первичной ориентации.

---

## 2) Если задача про legacy-конвейер (json sink, embedding, Postgres cache)

Читай в порядке:
1. `src/ollama_runner/pipeline.py`
2. `src/ollama_runner/process/parser.py`
3. `src/ollama_runner/process/resolver.py`
4. `src/ollama_runner/process/cache.py`
5. `src/ollama_runner/embeeding.py`

Связанные контракты:
- `src/ollama_runner/types.py`
- `src/ollama_runner/protocols.py`

Когда расширять область:
- если спорно, почему выбран конкретный `device_id`, смотри `src/ollama_runner/inventory/static.py`.

---

## 3) Если задача про semantic NLU и выбор устройства

Читай в порядке:
1. `src/ollama_runner/semantic_pipeline.py`
2. `src/ollama_runner/nlu/structure.py` (координация команд)
3. `src/ollama_runner/nlu/parse.py` (детерминированные intent-правила)
4. `src/ollama_runner/nlu/backend.py` (LLM fallback + schema)
5. `src/ollama_runner/resolve/capability.py`
6. `src/ollama_runner/resolve/active.py`

Доменные типы и статусы:
- `src/ollama_runner/semantic.py`
- `src/ollama_runner/inventory/registry.py`

Частая ошибка навигации: путать `content.play` (нужна конкретизация media type) с `audio.play`/`video.play`.

---

## 4) Если задача про интеграцию с Home Assistant

Читай в порядке:
1. `src/ollama_runner/main.py` (`run_home_request`)
2. `src/ollama_runner/ha/normalize.py` (discovery -> runtime inventory)
3. `src/ollama_runner/ha/route.py` (backend routing)
4. `src/ollama_runner/ha/execute.py` (service calls, failure reasons)
5. `src/ollama_runner/execution.py` (какие intents реально исполняются)
6. `src/ollama_runner/ha/client.py` (HTTP + websocket registry calls)

Если проблема выглядит как «capability есть, но не выполняется», первым делом проверь `execution.py`.

---

## 5) Если задача про инвентарь, owner/area/device_type

- Статический тестовый инвентарь: `src/ollama_runner/inventory/static.py`
- Канонизация и alias-логика: `src/ollama_runner/inventory/registry.py`
- Runtime-инвентарь из HA: `src/ollama_runner/ha/normalize.py`

Неочевидная связь: один и тот же resolver (`CapabilityResolver`) работает и со static, и с runtime registry.

---

## 6) Если задача про трассировку и диагностику запроса

- Формат результата запроса и агрегирование статусов: `src/ollama_runner/request.py`
- Семантические трассы pipeline: `src/ollama_runner/semantic_pipeline.py`
- Legacy trace mapping: `src/ollama_runner/skills/book.py`

---

## 7) Где лежат промпты

- Legacy parser prompt: путь из `Settings.prompt_path` (по умолчанию `src/ollama_runner/prompts/homeassist.md`)
- Semantic NLU prompt: `src/ollama_runner/prompts/semantic.md`

Меняя формат ответа модели, синхронизируй:
- schema в `process/parser.py` или `nlu/backend.py`;
- нормализацию downstream (`types.py`/`nlu/normalize.py`/`semantic_pipeline.py`).

---

## 8) Что обычно не нужно читать для прикладной правки

- `tests/**` (если задача не про тестовую инфраструктуру)
- корневые артефакты `*.test`, `*.jsonl`, `*.traces.json`
- `.pytest_cache/**`, `.venv/**`

Отдельно: файлы с расширением `*.test` не читать без прямого запроса пользователя.

---

## 9) Быстрые маршруты по типовым задачам

- Локальный баг разбора команды: `semantic_pipeline.py` -> `nlu/parse.py` -> `nlu/structure.py`
- Изменение публичного формата результата: `types.py` и/или `request.py` (+ call sites)
- Ошибка «между» NLU и HA: `resolve/capability.py` -> `ha/route.py` -> `ha/execute.py`
- Добавление новой выполняемой capability: `execution.py` -> `ha/execute.py` -> (при необходимости) `inventory/registry.py`
- Изменение переменных окружения/URL/таймаутов: `settings.py` -> потребители (`factory.py`, `sources/whisper.py`, `ha/client.py`)

---

## 10) Замеченные технические особенности

- В проекте есть исторические опечатки в именах модулей/скриптов: `embeeding.py`, `toogle_light.py`.
- Эти имена участвуют в `pyproject.toml` (`emb`, `toogle`), поэтому переименование затронет внешние точки входа.

---

## 11) Markdown-документы и их роль

- `README.md` — пользовательский и эксплуатационный контекст проекта.
- `docs/architecture.md` — архитектурная карта и связи подсистем.
- `docs/codebase-map.md` — навигационный индекс по задачам.
- `next.md` — рабочие исследовательские заметки по качеству семантики; не источник архитектурных контрактов.
