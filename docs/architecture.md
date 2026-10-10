# Архитектура ollama-runner

## Назначение проекта

`ollama-runner` — пайплайн разбора русскоязычных команд умного дома:

- вход: текстовый файл/строка или аудио (через Whisper);
- разбор: детерминированный NLU и/или LLM;
- резолвинг: выбор конкретного устройства по инвентарю и ограничениям;
- выход: legacy-команда (`Command`) и, в режиме HA, вызов Home Assistant service.

Проект поддерживает два операционных контура:

1. **Legacy pipeline** (бенч/совместимость): `Intent -> FastTextResolver -> Command`.
2. **Semantic pipeline** (текущий runtime): `SemanticCommand -> CapabilityResolver -> ExecutionRouter/HA`.

---

## Точки входа

- CLI: `src/ollama_runner/main.py` (`ollama-runner`)
  - `--sink json` → legacy pipeline (`factory.default_pipeline`)
  - `--sink ha` → semantic pipeline для Home Assistant (`run_home_request`)
- Утилита embed-резолвера: `src/ollama_runner/embeeding.py` (`emb`)
- Доп. утилита переключения света: `src/ollama_runner/toogle_light.py` (`toogle`)

Скрипты заданы в `pyproject.toml`.

---

## Подсистемы и ответственность

### 1) Оркестрация и фабрика

- `src/ollama_runner/factory.py`
- `src/ollama_runner/pipeline.py`

Что делает:
- собирает и кэширует ресурсы уровня процесса: Ollama-парсеры, NLU backend, PostgresCache, FastTextResolver;
- выдаёт настроенные пайплайны;
- закрывает shared-ресурсы (`close_resources`).

Ключевая связь: выбор модели влияет и на parser/NLU, и на namespace кэша (`model` в `command_cache`).

### 2) Источники входа

- `src/ollama_runner/sources/text.py`
- `src/ollama_runner/sources/whisper.py`

Что делает:
- преобразует input в `Utterance`.
- Whisper — HTTP-клиент к внешнему ASR (`WHISPER_*`).

### 3) Legacy parsing/resolving/cache

- Парсер LLM: `src/ollama_runner/process/parser.py` (`OllamaParser`, JSON schema `Intent`)
- Резолвер FastText: `src/ollama_runner/process/resolver.py`
- Кэш: `src/ollama_runner/process/cache.py` (`PostgresCache`, `MemoryCache`)
- Нормализация ключа кэша: `src/ollama_runner/textnorm.py`

Что делает:
- превращает текст в `Intent`;
- выбирает устройство через embedding по статическому inventory;
- кэширует `text_norm -> Command` в Postgres с учётом `model + inventory_v`.

### 4) Semantic NLU

- Оркестратор: `src/ollama_runner/semantic_pipeline.py`
- Детерминированный разбор и координация: `src/ollama_runner/nlu/parse.py`, `src/ollama_runner/nlu/structure.py`, `src/ollama_runner/nlu/slots.py`, `src/ollama_runner/nlu/values.py`
- LLM fallback backend: `src/ollama_runner/nlu/backend.py`
- Нормализация LLM-ответа: `src/ollama_runner/nlu/normalize.py`

Что делает:
- сначала пытается полностью разобрать фразу детерминированно;
- если не удалось — вызывает LLM со схемой `SEMANTIC_SCHEMA`;
- мержит explicit-слоты из текста поверх LLM-предсказания;
- формирует трассировку (`NLUTrace`, `observations`) и request-лог.

Критичный инвариант: **частично распарсенный coordinated plan не исполняется** как «понятый префикс»; unresolved-часть оставляет запрос в неуспехе.

### 5) Inventory / Registry / Capabilities

- Статический инвентарь: `src/ollama_runner/inventory/static.py`
- Реестр устройств и лингвистические канонизации: `src/ollama_runner/inventory/registry.py`
- Capability resolver: `src/ollama_runner/resolve/capability.py`
- Политики «активной сессии»: `src/ollama_runner/resolve/active.py`

Что делает:
- хранит канонические owner/area/device_type и capability-карту;
- фильтрует кандидатов hard constraints (owner/area/type/ordinal);
- учитывает relation-based выполнение (например, audio_output);
- ранжирует кандидатов и возвращает `resolved/ambiguous/not_found/unsupported/clarify`.

Неочевидное: resolver может вернуть fanout (несколько execution targets) для части интентов.

### 6) HA runtime: обнаружение и выполнение

- HA client (HTTP + raw WebSocket protocol): `src/ollama_runner/ha/client.py`
- Построение runtime-инвентаря: `src/ollama_runner/ha/normalize.py`
- Роутинг выполнения: `src/ollama_runner/ha/route.py`
- Реальное исполнение сервисов HA: `src/ollama_runner/ha/execute.py`
- Карта реализованных execution-capabilities: `src/ollama_runner/execution.py`

Что делает:
- читает актуальные entity/device/area/label/services/state из HA;
- строит `DeviceRegistry + Binding + DeviceRuntime` на лету для каждого запроса;
- выполняет только intents, объявленные в `EXECUTION_CAPABILITIES`;
- формирует детализированный статус выполнения (включая `execution_failed`).

Неочевидное:
- `load_inventory()` вызывается на каждый `run_home_request`; устаревшая карта не переиспользуется после ошибки обновления.
- Для registry API используется собственная WebSocket-реализация (без внешней ws-библиотеки).

### 7) Адаптация legacy-команд / sinks / трассировка

- Legacy mapping semantic->Command: `src/ollama_runner/skills/book.py`
- Sinks: `src/ollama_runner/sinks/*.py`
- Structured request trace/logging: `src/ollama_runner/request.py`

Что делает:
- обеспечивает совместимость с существующим форматом `Command`;
- централизует пользовательский trace запроса и редактирование чувствительных строк в логах.

---

## Направления зависимостей

Высокоуровнево:

`main.py`
→ `factory.py`
→ (`pipeline.py` или `semantic_pipeline.py`)
→ (`process/*` или `nlu/* + resolve/*`)
→ `skills/book.py`/`ha/*`
→ `sinks/*`

Конфигурация (`settings.py`) и доменные типы (`types.py`, `semantic.py`, `protocols.py`) используются поперечно.

---

## Основные потоки данных

### A. Legacy (`--sink json`)
1. Source читает `Utterance`.
2. Pipeline проверяет cache (`PostgresCache`).
3. При miss: `OllamaParser.parse -> Intent`.
4. `FastTextResolver.resolve -> Command`.
5. `PrintSink.send` печатает JSON.

### B. Home Assistant (`--sink ha`)
1. `run_home_request()` читает свежий HA inventory/state.
2. `SemanticPipeline.run(text)`:
   - deterministic plan; при полном разборе — без LLM;
   - иначе LLM fallback + merge explicit slots.
3. `CapabilityResolver.resolve` выбирает semantic/execution target.
4. `ExecutionRouter` выбирает backend и вызывает `HaExecutor`.
5. `HaExecutor` вызывает HA service или возвращает `unsupported/execution_failed`.
6. `request.emit_request` пишет компактный прод-трейс.

---

## Внешние интеграции

- **Ollama** (`/api/chat`) — два сценария:
  - legacy `Intent` parsing,
  - semantic NLU fallback.
- **Postgres** — cache таблица `command_cache`.
- **FastText model** (`cc.ru.300.bin`) — legacy embedding resolver.
- **Whisper HTTP** — транскрибация аудио.
- **Home Assistant**:
  - HTTP API (`/api/states`, `/api/services/...`),
  - WebSocket API (`/api/websocket`) для registry-команд.

---

## Ключевые контракты

- Legacy домен: `Intent`, `Command`, `Result` в `src/ollama_runner/types.py`.
- Semantic домен: `SemanticCommand`, `ResolvedCommand`, `SemanticNLUOutcome`, статусы (`resolved/ambiguous/not_found/unsupported/clarify`) в `src/ollama_runner/semantic.py`.
- Абстракции пайплайна: `Source/Parser/Resolver/Cache/Sink/Transcriber` в `src/ollama_runner/protocols.py`.

---

## Конфигурация и сборка

- Python `>=3.12`, сборка через `uv_build` (`pyproject.toml`).
- Основные переменные окружения задаются через `Settings` (`src/ollama_runner/settings.py`) и `.env`.
- В проекте нет сложной мульти-стейдж сборки; основная сложность — runtime-зависимости на внешние сервисы и модельные файлы.

---

## Риски при изменениях

1. **Смешение legacy и semantic контуров**: одинаковые термины, но разные контракты и статусные модели.
2. **Resolver/Execution граница**: advertised capability ≠ реализованная execution capability.
3. **HA discovery**: изменение нормализации alias/owner/area/ordinal влияет на выбор целей.
4. **Кэш**: изменение нормализации текста или inventory version ломает hit-rate/совместимость.
5. **LLM schema**: изменение enum/формата должно синхронно отражаться в normalizer и downstream.

---

## Что здесь точно отсутствует

- Нет Go/C/C++, FFI/CGO.
- Нет многопроцессной оркестрации и очередей.
- Нет отдельного web-сервера приложения.

Основная сложность — в правилах NLU/резолвинга и в runtime-интеграции с Home Assistant.
