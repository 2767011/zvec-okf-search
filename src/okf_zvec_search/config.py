from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

DIMENSION = 384


DEFAULT_MODEL = "e5"


DEFAULT_SEARCH_MODE = "hybrid"


SEARCH_MODES = ("semantic", "fts", "fts_raw", "hybrid")


APP_HOME = Path(os.environ.get("OKF_ZVEC_HOME", "/opt/okf-zvec-search"))


DEFAULT_OKF_DIR = APP_HOME / "data" / "okf"


DEFAULT_DB_ROOT = APP_HOME / "data" / "db"


MODEL_CONFIGS = {
    "paraphrase": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


_SERVICE_TOKEN_FILE = Path(os.environ.get("OKF_ZVEC_TOKEN_FILE", str(APP_HOME / "config" / "service-token")))


_SEARCH_TOKEN_FILE = Path(
    os.environ.get("OKF_ZVEC_SEARCH_TOKEN_FILE", str(APP_HOME / "config" / "search-token"))
)


_ADMIN_TOKEN_FILE = Path(
    os.environ.get("OKF_ZVEC_ADMIN_TOKEN_FILE", str(APP_HOME / "config" / "admin-token"))
)


_ACTIVE_DB_FILE = Path(os.environ.get("OKF_ZVEC_ACTIVE_DB_FILE", str(APP_HOME / "data" / "active-db-root")))


_RUNTIME_SETTINGS_FILE = Path(
    os.environ.get(
        "OKF_ZVEC_RUNTIME_SETTINGS_FILE",
        str(APP_HOME / "config" / "runtime-settings.json"),
    )
)


_AI_HISTORY_FILE = Path(
    os.environ.get(
        "OKF_ZVEC_AI_HISTORY_FILE",
        str(APP_HOME / "config" / "ai-search-history.json"),
    )
)


_AI_HISTORY_LIMIT = 20


DEFAULT_KEEP_VERSIONS = 3


DEFAULT_SEMANTIC_WEIGHT = 1.0


DEFAULT_FTS_WEIGHT = 1.0


DEFAULT_MIN_RELEVANCE = 0.25


RRF_RANK_CONSTANT = 60


DEFAULT_MAX_SYNC_BYTES = 100 * 1024 * 1024


DEFAULT_MAX_EXTRACTED_BYTES = 500 * 1024 * 1024


DEFAULT_MAX_ARCHIVE_MEMBERS = 10_000


DEFAULT_INDEX_BATCH_SIZE = 256


MAX_TOPK = 100


MAX_RERANK_POOL = 1_000


MAX_SNIPPET = 10_000


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchOptions:
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    fts_weight: float = DEFAULT_FTS_WEIGHT
    min_relevance: float = DEFAULT_MIN_RELEVANCE
    doc_type: str = ""
    tags: tuple[str, ...] = ()
    path_pattern: str = ""
    project: str = ""
    date_from: str = ""
    date_to: str = ""

    def validate(self) -> None:
        if self.semantic_weight < 0 or self.fts_weight < 0:
            raise ValueError("веса поиска не могут быть отрицательными")
        if self.semantic_weight + self.fts_weight <= 0:
            raise ValueError("хотя бы один вес поиска должен быть больше нуля")
        if not 0 <= self.min_relevance <= 1:
            raise ValueError("порог релевантности должен быть от 0 до 1")


def env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} должно быть числом") from exc


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} должно быть целым числом") from exc
    if value < minimum:
        raise ValueError(f"{name} должно быть не меньше {minimum}")
    return value


def bounded_int(
    value: str,
    name: str,
    minimum: int,
    maximum: int,
    overflow_status: int = 400,
) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RequestError(400, f"{name} должно быть целым числом") from exc
    if parsed > maximum:
        raise RequestError(overflow_status, f"{name} должно быть не больше {maximum}")
    if parsed < minimum:
        raise RequestError(400, f"{name} должно быть от {minimum} до {maximum}")
    return parsed


def split_filter_tags(value: str | None) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in (value or "").split(",") if tag.strip())


def make_search_options(
    *,
    semantic_weight: float | None = None,
    fts_weight: float | None = None,
    min_relevance: float | None = None,
    doc_type: str = "",
    tags: str = "",
    path_pattern: str = "",
    project: str = "",
    date_from: str = "",
    date_to: str = "",
) -> SearchOptions:
    options = SearchOptions(
        semantic_weight=(
            semantic_weight
            if semantic_weight is not None
            else env_float("OKF_ZVEC_SEMANTIC_WEIGHT", DEFAULT_SEMANTIC_WEIGHT)
        ),
        fts_weight=(
            fts_weight if fts_weight is not None else env_float("OKF_ZVEC_FTS_WEIGHT", DEFAULT_FTS_WEIGHT)
        ),
        min_relevance=(
            min_relevance
            if min_relevance is not None
            else env_float("OKF_ZVEC_MIN_RELEVANCE", DEFAULT_MIN_RELEVANCE)
        ),
        doc_type=doc_type.strip(),
        tags=split_filter_tags(tags),
        path_pattern=path_pattern.strip(),
        project=project.strip(),
        date_from=date_from.strip(),
        date_to=date_to.strip(),
    )
    options.validate()
    return options


def normalize_model_key(model_key: str | None) -> str:
    key = (model_key or DEFAULT_MODEL).casefold()
    if key not in MODEL_CONFIGS:
        raise ValueError(f"неизвестная модель '{model_key}'; доступно: {', '.join(MODEL_CONFIGS)}")
    return key


def normalize_search_mode(search_mode: str | None) -> str:
    mode = (search_mode or DEFAULT_SEARCH_MODE).casefold()
    if mode not in SEARCH_MODES:
        raise ValueError(f"неизвестный режим '{search_mode}'; доступно: {', '.join(SEARCH_MODES)}")
    return mode
