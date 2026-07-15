# OKF Zvec Search

Автономный сервис семантического, полнотекстового и гибридного поиска по
Markdown-базе знаний в формате
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/open-knowledge).

Сервис разбивает Markdown по заголовкам и элементам списков, предоставляет
веб-интерфейс и HTTP API. Он рассчитан на многоязычные базы знаний и использует
русскую лемматизацию для поиска BM25.

## Режимы поиска

| Режим | Механизм | Лучше всего подходит для |
| --- | --- | --- |
| `semantic` | многоязычные эмбеддинги и косинусная близость | синонимов и поиска по смыслу |
| `fts` | zvec BM25 по нормализованным леммам | точных терминов и быстрых ответов |
| `hybrid` | семантика и FTS, объединённые через RRF | повседневного использования |

Поддерживаемые модели эмбеддингов:

- `intfloat/multilingual-e5-small` по умолчанию;
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Требования

- Linux x86_64 или ARM64;
- Python 3.10-3.14;
- рекомендуется 4 ядра процессора;
- рекомендуется 8 ГБ оперативной памяти при одновременной загрузке двух моделей;
- не менее 10 ГБ свободного места для приложения, моделей и индексов.

## Быстрый запуск

```bash
git clone https://github.com/2767011/zvec-okf-search.git
cd zvec-okf-search
sudo ./deploy/install.sh
sudo systemctl enable --now okf-zvec-search
```

Загрузка каталога OKF из Windows:

```powershell
.\scripts\sync.ps1 `
  -OkfPath C:\путь\к\okf `
  -ServiceUrl http://АДРЕС_СЕРВЕРА:8765 `
  -TokenFile .\service-token
```

После этого откройте `http://АДРЕС_СЕРВЕРА:8765/`.

Подробная установка, настройка, проверка и диагностика описаны в
[ONBOARDING.md](ONBOARDING.md).

## HTTP API

```text
GET /health
GET /models
GET /status
GET /status.json
GET /metrics
GET /search?q=портал&topk=5&model=e5&mode=hybrid
POST /sync
```

`POST /sync` принимает сжатый gzip tar-архив с каталогом `okf` в корне.
Для запроса нужен токен сервиса в заголовке `X-OKF-Zvec-Token`.

Пример:

```bash
curl --get \
  --data-urlencode "q=перенос телефонии" \
  --data "topk=5" \
  --data "model=e5" \
  --data "mode=hybrid" \
  http://127.0.0.1:8765/search
```

## Командная строка

```bash
okf-zvec index --okf ./examples/okf --model all
okf-zvec search "портал поставщиков" --model e5 --mode hybrid
okf-zvec serve --okf ./examples/okf --host 0.0.0.0 --port 8765
```

Пути конфигурации задаются переменными окружения:

```text
OKF_ZVEC_HOME
OKF_ZVEC_TOKEN_FILE
OKF_ZVEC_ACTIVE_DB_FILE
OKF_ZVEC_KEEP_VERSIONS
OKF_ZVEC_SEARCH_TOKEN_FILE
OKF_ZVEC_PRELOAD_MODELS
HF_TOKEN
```

`OKF_ZVEC_KEEP_VERSIONS` задаёт число сохраняемых версий индекса и по умолчанию
равно `3`. Новая версия становится активной только после успешной сборки всех
моделей. При ошибке сервис продолжает использовать прежние OKF и индекс.

Веса гибридного поиска и порог выдачи задаются через
`OKF_ZVEC_SEMANTIC_WEIGHT`, `OKF_ZVEC_FTS_WEIGHT` и
`OKF_ZVEC_MIN_RELEVANCE`. Порог находится в диапазоне от `0` до `1`.

Ограничения синхронизации задаются переменными
`OKF_ZVEC_MAX_SYNC_BYTES`, `OKF_ZVEC_MAX_EXTRACTED_BYTES` и
`OKF_ZVEC_MAX_ARCHIVE_MEMBERS`. Размер партии индексации управляется через
`OKF_ZVEC_INDEX_BATCH_SIZE`.

Фильтры доступны в HTTP API, CLI и веб-интерфейсе:

```bash
okf-zvec search "миграция" \
  --mode hybrid \
  --type software-project \
  --tags zvec,okf \
  --path "topics/*" \
  --project search \
  --date-from 2026-07-01 \
  --min-relevance 0.35 \
  --semantic-weight 1.5 \
  --fts-weight 1
```

Каждый результат содержит нормализованную `relevance`, использованные сигналы,
причину попадания в выдачу и словоформы для подсветки.

## Авторизация поиска

Если файл `OKF_ZVEC_SEARCH_TOKEN_FILE` существует и содержит токен,
веб-интерфейс, `/search`, `/status`, `/models` и `/metrics` требуют
авторизацию. `/health` остаётся открытым.

Браузер показывает стандартное окно входа:

- имя пользователя: `okf`;
- пароль: содержимое файла поискового токена.

API принимает Basic Auth, Bearer или заголовок
`X-OKF-Zvec-Search-Token`. PowerShell-клиент принимает `-TokenFile`:

```powershell
.\scripts\search.ps1 "миграция" `
  -ServiceUrl http://SERVER_IP:8765 `
  -TokenFile .\search-token
```

Если файл токена отсутствует, поиск закрывается с HTTP 401. Анонимный режим
можно включить только явно: `OKF_ZVEC_ALLOW_ANONYMOUS_SEARCH=1`.

Управляющие запросы на загрузку моделей, изменение автозагрузки и перезапуск
защищены отдельным файлом `OKF_ZVEC_ADMIN_TOKEN_FILE`. При первом действии на
странице состояния браузер запрашивает admin-токен и хранит его до закрытия
вкладки в `sessionStorage`.

## Загрузка моделей

Предзагрузка моделей управляется через `OKF_ZVEC_PRELOAD_MODELS`. В
systemd-конфигурации по умолчанию сразу загружается основная модель E5, поэтому
первый semantic или hybrid-запрос выполняется без задержки на её запуск:

```text
OKF_ZVEC_PRELOAD_MODELS=e5
OKF_ZVEC_PRELOAD_MODELS=e5,paraphrase
OKF_ZVEC_PRELOAD_MODELS=all
```

Пустое значение включает полностью ленивую загрузку. `all` загружает обе
модели. FTS не требует embedding-модели.

На странице `/status` можно отдельно управлять моделями в памяти: загрузить,
выгрузить или перезагрузить каждую из них. Чекбоксы автозагрузки сохраняются в
`OKF_ZVEC_RUNTIME_SETTINGS_FILE`, применяются сразу и имеют приоритет над
переменной окружения. Перезапуск сервиса выполняется там же.

Если сервис полностью остановлен, его запуск выполняется на сервере:

```bash
sudo systemctl start okf-zvec-search
```

## Состояние, журналы и метрики

- `/status` — страница состояния;
- `/status.json` — состояние в JSON;
- `/metrics` — метрики Prometheus.

Отдельная страница `/ai-history` показывает последние 20 запросов, помеченных как обращения
от ИИ. Для каждого сохраняются длительность, количество результатов, лучший
документ и его релевантность. Над таблицей рассчитывается сводка по ошибкам,
пустым выдачам и средним значениям. История переживает перезапуск сервиса и
хранится в `OKF_ZVEC_AI_HISTORY_FILE`.

AI-клиент должен передать заголовок `X-OKF-Zvec-Origin: ai`. Обычные браузерные
и API-запросы в эту историю не входят. PowerShell-клиент поддерживает параметр:

```powershell
.\scripts\search.ps1 "миграция" -Origin ai
```

Сервис пишет в stdout однострочные JSON-события: запуск, загрузка модели,
поиск и синхронизация. Текст поискового запроса в журнал не записывается.

Пример Prometheus:

```yaml
scrape_configs:
  - job_name: okf-zvec-search
    basic_auth:
      username: okf
      password_file: /secure/search-token
    static_configs:
      - targets: ["SERVER_IP:8765"]
```

## Проверка качества

Публичный корпус находится в `benchmarks/corpus`, а 50 контрольных запросов —
в `benchmarks/queries.json`. Команда сравнивает `semantic`, `fts` и `hybrid`,
рассчитывая Top-1, Top-3, MRR, Recall@k, nDCG@k и среднюю задержку:

```bash
okf-zvec benchmark \
  --file benchmarks/queries.json \
  --service-url http://127.0.0.1:8765 \
  --token-file ./search-token \
  --model e5 \
  --modes semantic,fts,hybrid
```

## Безопасность

- Не добавляйте в Git токены сервиса и Hugging Face, закрытые базы OKF и
  сгенерированные индексы.
- История ИИ содержит тексты последних запросов и доступна пользователям
  защищённой страницы истории; учитывайте это при выдаче поискового токена.
- Ограничьте доступ к сервису межсетевым экраном или обратным прокси.
  Если настроен поисковый токен, веб-интерфейс, поиск, состояние и метрики
  требуют авторизацию. Синхронизация защищена отдельным сервисным токеном.
- Для доступа через недоверенную сеть используйте HTTPS: Basic- и
  Bearer-аутентификация сами по себе не шифруют токены.
- При загрузке архива сервис проверяет пути и использует безопасный фильтр
  распаковки Python.
- Служба systemd работает от непривилегированного пользователя `okf-zvec` и
  имеет право записи только в каталоги `config` и `data`.

## Разработка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

В PowerShell окружение активируется командой
`.\.venv\Scripts\Activate.ps1`.

## Лицензия

MIT
