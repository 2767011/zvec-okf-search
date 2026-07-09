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
git clone https://github.com/YOUR_ACCOUNT/okf-zvec-search.git
cd okf-zvec-search
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
HF_TOKEN
```

## Безопасность

- Не добавляйте в Git токены сервиса и Hugging Face, закрытые базы OKF и
  сгенерированные индексы.
- Ограничьте доступ к сервису межсетевым экраном или обратным прокси.
  Эндпоинты поиска намеренно не требуют авторизации, а синхронизация защищена
  токеном.
- При загрузке архива сервис проверяет пути и использует безопасный фильтр
  распаковки Python.

## Разработка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

## Лицензия

MIT
