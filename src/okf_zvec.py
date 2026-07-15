#!/usr/bin/env python3
"""Локальный поиск по OKF на базе zvec."""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict, deque
import contextlib
import fnmatch
import hashlib
import html
import json
import math
import os
import re
import secrets
import shutil
import sys
import tarfile
import threading
import tempfile
import gc
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import zvec
import yaml


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
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"[\wА-Яа-яЁё-]+", re.UNICODE)
_MODELS: dict[str, Any] = {}
_SEARCH_COLLECTIONS: dict[str, Any] = {}
_SEARCH_LOCK = threading.Lock()
_MODEL_LOCK = threading.RLock()
_MODEL_INFERENCE_LOCKS = {model_key: threading.Lock() for model_key in MODEL_CONFIGS}
_SYNC_LOCK = threading.Lock()
_SYNC_IN_PROGRESS = threading.Event()
_RESTART_REQUESTED = threading.Event()
_QUERY_CACHE_LOCK = threading.Lock()
_QUERY_CACHE: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
_QUERY_CACHE_MAX = 256
_SERVICE_TOKEN_FILE = Path(
    os.environ.get("OKF_ZVEC_TOKEN_FILE", str(APP_HOME / "config" / "service-token"))
)
_SEARCH_TOKEN_FILE = Path(
    os.environ.get("OKF_ZVEC_SEARCH_TOKEN_FILE", str(APP_HOME / "config" / "search-token"))
)
_ADMIN_TOKEN_FILE = Path(
    os.environ.get("OKF_ZVEC_ADMIN_TOKEN_FILE", str(APP_HOME / "config" / "admin-token"))
)
_ACTIVE_DB_FILE = Path(
    os.environ.get("OKF_ZVEC_ACTIVE_DB_FILE", str(APP_HOME / "data" / "active-db-root"))
)
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
_AI_HISTORY_LOCK = threading.Lock()
_AI_HISTORY: deque[dict[str, Any]] = deque(maxlen=_AI_HISTORY_LIMIT)
_AI_HISTORY_LOADED = False
_MORPH: Any | None = None
_MORPH_UNAVAILABLE = False
DEFAULT_KEEP_VERSIONS = 3
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_FTS_WEIGHT = 1.0
DEFAULT_MIN_RELEVANCE = 0.25
RRF_RANK_CONSTANT = 60
SERVICE_STARTED_AT = time.time()
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


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


class ServiceMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.search_requests: dict[tuple[str, str, str], int] = {}
        self.search_duration_sum: dict[tuple[str, str], float] = {}
        self.search_duration_count: dict[tuple[str, str], int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.sync_total: dict[str, int] = {"success": 0, "error": 0}
        self.sync_duration_sum = 0.0
        self.sync_duration_count = 0
        self.model_load_seconds: dict[str, float] = {}

    def record_search(self, model: str, mode: str, status: str, duration: float) -> None:
        with self.lock:
            status_key = (model, mode, status)
            duration_key = (model, mode)
            self.search_requests[status_key] = self.search_requests.get(status_key, 0) + 1
            self.search_duration_sum[duration_key] = (
                self.search_duration_sum.get(duration_key, 0.0) + duration
            )
            self.search_duration_count[duration_key] = (
                self.search_duration_count.get(duration_key, 0) + 1
            )

    def record_cache(self, hit: bool) -> None:
        with self.lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def record_sync(self, status: str, duration: float) -> None:
        with self.lock:
            self.sync_total[status] = self.sync_total.get(status, 0) + 1
            self.sync_duration_sum += duration
            self.sync_duration_count += 1

    def record_model_load(self, model: str, duration: float) -> None:
        with self.lock:
            self.model_load_seconds[model] = duration


_METRICS = ServiceMetrics()
_STATE_LOCK = threading.Lock()
_SERVICE_STATE: dict[str, Any] = {
    "last_sync_at": "",
    "last_sync_status": "never",
    "last_sync_duration_seconds": 0.0,
    "last_sync_error": "",
    "active_db_root": "",
    "models": {},
}


def normalize_preload_setting(value: str) -> tuple[str, list[str]]:
    normalized = value.strip().casefold()
    if normalized in ("", "none"):
        return "none", []
    if normalized in ("1", "true", "yes", "all"):
        return "all", list(MODEL_CONFIGS)
    models = list(dict.fromkeys(
        normalize_model_key(item.strip()) for item in normalized.split(",") if item.strip()
    ))
    canonical = ",".join(models)
    return canonical, models


def configured_index_models() -> list[str]:
    value = os.environ.get("OKF_ZVEC_INDEX_MODELS", "all")
    _, models = normalize_preload_setting(value)
    if not models:
        raise ValueError("OKF_ZVEC_INDEX_MODELS должен содержать хотя бы одну модель")
    return models


def configured_preload_setting() -> tuple[str, list[str]]:
    try:
        payload = json.loads(_RUNTIME_SETTINGS_FILE.read_text(encoding="utf-8"))
        value = str(payload.get("preload_models", ""))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        value = os.environ.get("OKF_ZVEC_PRELOAD_MODELS", "")
    return normalize_preload_setting(value)


def save_preload_setting(value: str) -> tuple[str, list[str]]:
    canonical, models = normalize_preload_setting(value)
    _RUNTIME_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = _RUNTIME_SETTINGS_FILE.with_name(
        f".{_RUNTIME_SETTINGS_FILE.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps({"preload_models": canonical}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, _RUNTIME_SETTINGS_FILE)
    return canonical, models


def apply_preload_setting(value: str, unload_others: bool = True) -> tuple[str, list[str]]:
    canonical, models = normalize_preload_setting(value)
    unavailable = [model for model in models if model not in configured_index_models()]
    if unavailable:
        raise ValueError("предзагрузка требует включённого индекса: " + ", ".join(unavailable))
    for model_key in models:
        get_model(model_key)
    with _SEARCH_LOCK:
        with _MODEL_LOCK:
            if unload_others:
                for model_key in list(_MODELS):
                    if model_key not in models:
                        _MODELS.pop(model_key, None)
                gc.collect()
            loaded_models = sorted(_MODELS)
        with _STATE_LOCK:
            for model_key in MODEL_CONFIGS:
                state = _SERVICE_STATE["models"].setdefault(
                    model_key,
                    {"name": MODEL_CONFIGS[model_key]["name"]},
                )
                state["loaded"] = model_key in loaded_models
    log_event("preload_setting_applied", value=canonical, loaded_models=loaded_models)
    return canonical, models


def reload_preloaded_models() -> tuple[str, list[str]]:
    canonical, models = configured_preload_setting()
    with _SEARCH_LOCK:
        with _MODEL_LOCK:
            for model_key in models:
                _MODELS.pop(model_key, None)
        gc.collect()
    for model_key in models:
        get_model(model_key)
    with _MODEL_LOCK:
        loaded_models = sorted(_MODELS)
    log_event("preloaded_models_reloaded", value=canonical, loaded_models=loaded_models)
    return canonical, models


def update_loaded_model_state() -> list[str]:
    with _MODEL_LOCK:
        loaded_models = sorted(_MODELS)
    with _STATE_LOCK:
        for model_key in MODEL_CONFIGS:
            state = _SERVICE_STATE["models"].setdefault(
                model_key,
                {"name": MODEL_CONFIGS[model_key]["name"]},
            )
            state["loaded"] = model_key in loaded_models
    return loaded_models


def load_model(model_key: str) -> None:
    model_key = normalize_model_key(model_key)
    get_model(model_key)
    loaded_models = update_loaded_model_state()
    log_event("model_loaded_by_action", model=model_key, loaded_models=loaded_models)


def unload_model(model_key: str) -> None:
    model_key = normalize_model_key(model_key)
    with _SEARCH_LOCK:
        with _MODEL_LOCK:
            _MODELS.pop(model_key, None)
        gc.collect()
    loaded_models = update_loaded_model_state()
    log_event("model_unloaded_by_action", model=model_key, loaded_models=loaded_models)


def reload_model(model_key: str) -> None:
    unload_model(model_key)
    load_model(model_key)


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
            fts_weight
            if fts_weight is not None
            else env_float("OKF_ZVEC_FTS_WEIGHT", DEFAULT_FTS_WEIGHT)
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


def service_default_model() -> str:
    if DEFAULT_MODEL in _SEARCH_COLLECTIONS:
        return DEFAULT_MODEL
    return next(iter(_SEARCH_COLLECTIONS), DEFAULT_MODEL)


def get_model(model_key: str = DEFAULT_MODEL) -> Any:
    model_key = normalize_model_key(model_key)
    with _MODEL_LOCK:
        if model_key in _MODELS:
            return _MODELS[model_key]

        started = time.perf_counter()
        log_event("model_load_started", model=model_key, name=MODEL_CONFIGS[model_key]["name"])
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelLoadError(
                "Для семантического поиска требуется sentence-transformers. "
                "Установите зависимости проекта перед запуском сервиса."
            ) from exc

        try:
            model = SentenceTransformer(
                MODEL_CONFIGS[model_key]["name"],
                device="cpu",
                local_files_only=True,
            )
        except Exception:
            try:
                model = SentenceTransformer(MODEL_CONFIGS[model_key]["name"], device="cpu")
            except Exception as exc:
                raise ModelLoadError(f"не удалось загрузить модель {model_key}") from exc
        if hasattr(model, "get_embedding_dimension"):
            actual_dimension = model.get_embedding_dimension()
        else:
            actual_dimension = model.get_sentence_embedding_dimension()
        if actual_dimension != DIMENSION:
            raise ModelLoadError(
                f"Модель {model_key} возвращает {actual_dimension} измерений вместо {DIMENSION}."
            )
        _MODELS[model_key] = model
        duration = time.perf_counter() - started
        _METRICS.record_model_load(model_key, duration)
        with _STATE_LOCK:
            model_state = _SERVICE_STATE["models"].setdefault(model_key, {})
            model_state.update({
                "name": MODEL_CONFIGS[model_key]["name"],
                "loaded": True,
                "load_seconds": round(duration, 3),
            })
        log_event(
            "model_load_completed",
            model=model_key,
            name=MODEL_CONFIGS[model_key]["name"],
            duration_seconds=round(duration, 3),
        )
        return model


def prefixed_text(text: str, model_key: str, kind: str) -> str:
    config = MODEL_CONFIGS[normalize_model_key(model_key)]
    prefix = config["query_prefix"] if kind == "query" else config["passage_prefix"]
    return f"{prefix}{text}"


def embed(text: str, model_key: str = DEFAULT_MODEL, kind: str = "query") -> list[float]:
    model_key = normalize_model_key(model_key)
    model = get_model(model_key)
    with _MODEL_INFERENCE_LOCKS[model_key]:
        vector = model.encode(
            prefixed_text(text, model_key, kind),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return vector.astype("float32").tolist()


def embed_many(texts: list[str], model_key: str = DEFAULT_MODEL, kind: str = "passage") -> list[list[float]]:
    if not texts:
        return []
    model_key = normalize_model_key(model_key)
    model = get_model(model_key)
    with _MODEL_INFERENCE_LOCKS[model_key]:
        vectors = model.encode(
            [prefixed_text(text, model_key, kind) for text in texts],
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
    return [vector.astype("float32").tolist() for vector in vectors]


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    loaded = yaml.safe_load(match.group(1)) or {}
    meta = loaded if isinstance(loaded, dict) else {}
    return meta, text[match.end() :]


def metadata_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def metadata_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [metadata_text(item) for item in value if metadata_text(item)]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        return [item.strip().strip("'\"") for item in stripped.split(",") if item.strip()]
    return []


def title_from_markdown(body: str, fallback: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def split_block_by_bullets(block: str) -> list[str]:
    lines = block.splitlines()
    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("- "):
            if current:
                chunks.append("\n".join(current).strip())
            current = [line]
        elif current:
            current.append(line)
        elif line.strip():
            chunks.append(line.strip())

    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def section_chunks(body: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("", stripped)] if stripped else []

    chunks: list[tuple[str, str]] = []
    preface = body[: matches[0].start()].strip()
    if preface:
        chunks.append(("", preface))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        heading = match.group(2).strip()
        heading_line = match.group(0).strip()
        block = body[start:end].strip()
        if not block:
            chunks.append((heading, heading_line))
            continue

        bullet_chunks = split_block_by_bullets(block)
        if len(bullet_chunks) <= 1:
            chunks.append((heading, f"{heading_line}\n\n{block}".strip()))
        else:
            for bullet in bullet_chunks:
                chunks.append((heading, f"{heading_line}\n\n{bullet}".strip()))
    return chunks


def iter_doc_items(okf_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(okf_dir.rglob("*.md")):
        rel = path.relative_to(okf_dir).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"файл OKF должен быть в UTF-8: {rel}") from exc
        meta, body = split_frontmatter(text)
        title = metadata_text(meta.get("title")) or title_from_markdown(body, rel)
        doc_type = metadata_text(meta.get("type"))
        tags = metadata_tags(meta.get("tags"))
        project = metadata_text(meta.get("project"))
        timestamp = metadata_text(meta.get("timestamp"))
        for chunk_index, (heading, chunk) in enumerate(section_chunks(body)):
            safe_id = "doc_" + hashlib.sha1(f"{rel}#{chunk_index}".encode("utf-8")).hexdigest()
            search_text = f"{title}\n{doc_type}\n{' '.join(tags)}\n{project}\n{heading}\n{chunk}"
            yield {
                "id": safe_id,
                "path": rel,
                "chunk": str(chunk_index),
                "title": title,
                "type": doc_type,
                "tags": tags,
                "project": project,
                "timestamp": timestamp,
                "heading": heading,
                "text": chunk,
                "search_text": search_text,
                "fts_text": normalize_fts_text(search_text),
                "fts_raw_text": normalize_raw_fts_text(search_text),
            }


def docs_from_items(items: list[dict[str, Any]], model_key: str) -> list[zvec.Doc]:
    docs: list[zvec.Doc] = []
    for item, vector in zip(
        items, embed_many([item["search_text"] for item in items], model_key, kind="passage"), strict=True
    ):
        docs.append(
            zvec.Doc(
                id=item["id"],
                vectors={"embedding": vector},
                fields={
                    "path": item["path"],
                    "chunk": item["chunk"],
                    "heading": item["heading"],
                    "title": item["title"],
                    "type": item["type"],
                    "tags": item["tags"],
                    "project": item["project"],
                    "timestamp": item["timestamp"],
                    "filter_path": item["path"].casefold(),
                    "filter_type": item["type"].casefold(),
                    "filter_tags": [str(tag).casefold() for tag in item["tags"]],
                    "filter_project": item["project"].casefold(),
                    "filter_timestamp": item["timestamp"],
                    "text": item["text"],
                    "search_text": item["fts_text"],
                    "search_text_raw": item["fts_raw_text"],
                },
            )
        )
    return docs


def iter_docs(okf_dir: Path, model_key: str = DEFAULT_MODEL) -> list[zvec.Doc]:
    return docs_from_items(list(iter_doc_items(okf_dir)), model_key)


def create_schema() -> zvec.CollectionSchema:
    return zvec.CollectionSchema(
        name="okf_memory",
        fields=[
            zvec.FieldSchema("path", zvec.DataType.STRING),
            zvec.FieldSchema("chunk", zvec.DataType.STRING),
            zvec.FieldSchema("heading", zvec.DataType.STRING),
            zvec.FieldSchema("title", zvec.DataType.STRING),
            zvec.FieldSchema("type", zvec.DataType.STRING),
            zvec.FieldSchema("tags", zvec.DataType.ARRAY_STRING),
            zvec.FieldSchema("project", zvec.DataType.STRING),
            zvec.FieldSchema("timestamp", zvec.DataType.STRING),
            zvec.FieldSchema(
                "filter_path",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.InvertIndexParam(enable_extended_wildcard=True),
            ),
            zvec.FieldSchema(
                "filter_type",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.InvertIndexParam(),
            ),
            zvec.FieldSchema(
                "filter_tags",
                zvec.DataType.ARRAY_STRING,
                nullable=False,
                index_param=zvec.InvertIndexParam(),
            ),
            zvec.FieldSchema(
                "filter_project",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.InvertIndexParam(),
            ),
            zvec.FieldSchema(
                "filter_timestamp",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema(
                "search_text",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.FtsIndexParam(tokenizer_name="whitespace", filters=["lowercase"]),
            ),
            zvec.FieldSchema(
                "search_text_raw",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.FtsIndexParam(tokenizer_name="whitespace", filters=["lowercase"]),
            ),
            zvec.FieldSchema(
                "text",
                zvec.DataType.STRING,
                nullable=False,
            ),
        ],
        vectors=zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP32,
            DIMENSION,
            index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE),
        ),
    )


def doc_fields(result: Any) -> dict[str, Any]:
    fields = getattr(result, "fields", None)
    if isinstance(fields, dict):
        return fields
    if isinstance(result, dict):
        return result.get("fields", {})
    return {}


def doc_id(result: Any) -> str:
    return str(getattr(result, "id", None) or result.get("id", ""))


def doc_score(result: Any) -> float:
    return float(getattr(result, "score", None) if hasattr(result, "score") else result.get("score", 0.0))


def get_morph() -> Any | None:
    global _MORPH, _MORPH_UNAVAILABLE
    if _MORPH is not None:
        return _MORPH
    if _MORPH_UNAVAILABLE:
        return None
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError:
        _MORPH_UNAVAILABLE = True
        return None
    _MORPH = MorphAnalyzer()
    return _MORPH


@lru_cache(maxsize=8192)
def token_lemma(token: str) -> str:
    normalized = token.casefold()
    morph = get_morph()
    if morph is None or not re.search(r"[а-яё]", normalized):
        return normalized
    return morph.parse(normalized)[0].normal_form


def lexical_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if len(token) >= 3
    }


def normalize_fts_text(text: str) -> str:
    return " ".join(token_lemma(token) for token in TOKEN_RE.findall(text.casefold()))


def normalize_raw_fts_text(text: str) -> str:
    return " ".join(token.casefold() for token in TOKEN_RE.findall(text))


def result_matches_filters(fields: dict[str, Any], options: SearchOptions) -> bool:
    if options.doc_type and str(fields.get("type", "")).casefold() != options.doc_type.casefold():
        return False
    if options.project and str(fields.get("project", "")).casefold() != options.project.casefold():
        return False
    if options.path_pattern and not fnmatch.fnmatch(
        str(fields.get("path", "")).casefold(),
        options.path_pattern.casefold(),
    ):
        return False

    field_tags = {str(tag).casefold() for tag in (fields.get("tags") or [])}
    if options.tags and not all(tag.casefold() in field_tags for tag in options.tags):
        return False

    timestamp = str(fields.get("timestamp", ""))
    timestamp_from = timestamp[:10] if len(options.date_from) == 10 else timestamp
    timestamp_to = timestamp[:10] if len(options.date_to) == 10 else timestamp
    if options.date_from and (not timestamp or timestamp_from < options.date_from):
        return False
    if options.date_to and (not timestamp or timestamp_to > options.date_to):
        return False
    return True


def zvec_string(value: str) -> str:
    """Возвращает безопасный строковый литерал для выражения фильтра Zvec."""
    return json.dumps(value, ensure_ascii=False)


def zvec_path_pattern(pattern: str) -> str:
    """Преобразует обычную маску пути в шаблон LIKE Zvec."""
    normalized = pattern.casefold()
    if any(character in normalized for character in "[]%_"):
        raise ValueError("маска пути поддерживает только символы * и ?")
    return normalized.replace("*", "%").replace("?", "_")


def build_zvec_filter(options: SearchOptions) -> str | None:
    expressions: list[str] = []
    if options.doc_type:
        expressions.append(f"filter_type = {zvec_string(options.doc_type.casefold())}")
    if options.project:
        expressions.append(f"filter_project = {zvec_string(options.project.casefold())}")
    if options.path_pattern:
        if "*" in options.path_pattern or "?" in options.path_pattern:
            expressions.append(
                f"filter_path LIKE {zvec_string(zvec_path_pattern(options.path_pattern))}"
            )
        else:
            expressions.append(f"filter_path = {zvec_string(options.path_pattern.casefold())}")
    if options.tags:
        tags = ", ".join(zvec_string(tag.casefold()) for tag in options.tags)
        expressions.append(f"filter_tags CONTAIN_ALL ({tags})")
    if options.date_from:
        expressions.append("filter_timestamp != \"\"")
        expressions.append(f"filter_timestamp >= {zvec_string(options.date_from)}")
    if options.date_to:
        expressions.append("filter_timestamp != \"\"")
        upper_bound = options.date_to + "T23:59:59" if len(options.date_to) == 10 else options.date_to
        expressions.append(f"filter_timestamp <= {zvec_string(upper_bound)}")
    return " AND ".join(expressions) or None


def semantic_relevance(score: float) -> float:
    return max(0.0, min(1.0, 1.0 - score))


def fts_relevance(score: float) -> float:
    positive = max(0.0, score)
    return positive / (1.0 + positive)


def matching_terms(query: str, fields: dict[str, Any]) -> list[str]:
    query_lemmas = {token_lemma(token) for token in lexical_tokens(query)}
    if not query_lemmas:
        return []
    text = "\n".join(
        str(fields.get(name, "")) for name in ("title", "heading", "path", "text")
    )
    matched: dict[str, str] = {}
    for token in TOKEN_RE.findall(text):
        normalized = token.casefold()
        if token_lemma(normalized) in query_lemmas:
            matched.setdefault(normalized, token)
    return sorted(matched.values(), key=str.casefold)


def weighted_rrf(
    semantic_results: list[Any],
    fts_results: list[Any],
    options: SearchOptions,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    branches = (
        ("semantic", semantic_results, options.semantic_weight),
        ("fts", fts_results, options.fts_weight),
    )
    for signal, results, weight in branches:
        if weight <= 0:
            continue
        for rank, result in enumerate(results, start=1):
            identifier = doc_id(result)
            item = fused.setdefault(
                identifier,
                {
                    "result": result,
                    "rrf_score": 0.0,
                    "semantic_score": None,
                    "fts_score": None,
                    "signals": [],
                },
            )
            item["rrf_score"] += weight / (RRF_RANK_CONSTANT + rank)
            item[f"{signal}_score"] = doc_score(result)
            item["signals"].append(signal)
    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def hybrid_relevance(item: dict[str, Any], options: SearchOptions) -> float:
    signals: list[tuple[float, float]] = []
    if item["semantic_score"] is not None and options.semantic_weight > 0:
        signals.append((options.semantic_weight, semantic_relevance(item["semantic_score"])))
    if item["fts_score"] is not None and options.fts_weight > 0:
        signals.append((options.fts_weight, fts_relevance(item["fts_score"])))
    total_weight = sum(weight for weight, _ in signals)
    if total_weight == 0:
        return 0.0
    return sum(weight * relevance for weight, relevance in signals) / total_weight


def result_reason(signals: list[str], terms: list[str]) -> str:
    if signals == ["semantic"]:
        return "Семантическая близость запроса и фрагмента."
    if signals == ["fts"]:
        return f"Совпали термины: {', '.join(terms)}." if terms else "Полнотекстовое совпадение."
    if terms:
        return f"Семантическая близость и термины: {', '.join(terms)}."
    return "Результат объединённого семантического и полнотекстового поиска."


def clear_query_cache() -> None:
    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE.clear()


def search_collection(
    collection: Any,
    query: str,
    topk: int,
    rerank_pool: int,
    model_key: str = DEFAULT_MODEL,
    search_mode: str = DEFAULT_SEARCH_MODE,
    options: SearchOptions | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    if not query:
        raise ValueError("для поиска нужен непустой запрос")

    model_key = normalize_model_key(model_key)
    search_mode = normalize_search_mode(search_mode)
    options = options or SearchOptions()
    options.validate()
    cache_key = (
        model_key,
        search_mode,
        query.casefold().strip(),
        topk,
        rerank_pool,
        options,
    )
    if use_cache:
        with _QUERY_CACHE_LOCK:
            cached = _QUERY_CACHE.get(cache_key)
            if cached is not None:
                _QUERY_CACHE.move_to_end(cache_key)
                _METRICS.record_cache(True)
                return cached
        _METRICS.record_cache(False)

    pool_size = max(topk, rerank_pool)
    filter_expression = build_zvec_filter(options)
    semantic_results: list[Any] = []
    fts_results: list[Any] = []
    vector_query: Any | None = None
    if search_mode in ("semantic", "hybrid") and options.semantic_weight > 0:
        vector_query = zvec.Query("embedding", vector=embed(query, model_key, kind="query"))
    with _SEARCH_LOCK:
        if vector_query is not None:
            semantic_results = collection.query(
                vector_query,
                topk=pool_size,
                filter=filter_expression,
            )
        if search_mode in ("fts", "fts_raw", "hybrid") and options.fts_weight > 0:
            raw_fts = search_mode == "fts_raw"
            fts_query = zvec.Query(
                "search_text_raw" if raw_fts else "search_text",
                fts=zvec.Fts(
                    match_string=(
                        normalize_raw_fts_text(query) if raw_fts else normalize_fts_text(query)
                    )
                ),
            )
            fts_results = collection.query(
                fts_query,
                topk=pool_size,
                filter=filter_expression,
            )

    semantic_results = [
        result for result in semantic_results if result_matches_filters(doc_fields(result), options)
    ]
    fts_results = [
        result for result in fts_results if result_matches_filters(doc_fields(result), options)
    ]

    ranked: list[dict[str, Any]]
    if search_mode == "semantic":
        ranked = [
            {
                "result": result,
                "score": doc_score(result),
                "relevance": semantic_relevance(doc_score(result)),
                "semantic_score": doc_score(result),
                "fts_score": None,
                "signals": ["semantic"],
            }
            for result in semantic_results
        ]
    elif search_mode in ("fts", "fts_raw"):
        ranked = [
            {
                "result": result,
                "score": doc_score(result),
                "relevance": fts_relevance(doc_score(result)),
                "semantic_score": None,
                "fts_score": doc_score(result),
                "signals": ["fts"],
            }
            for result in fts_results
        ]
    else:
        ranked = weighted_rrf(semantic_results, fts_results, options)
        for item in ranked:
            item["score"] = item["rrf_score"]
            item["relevance"] = hybrid_relevance(item, options)

    output: list[dict[str, Any]] = []
    for item in ranked:
        if item["relevance"] < options.min_relevance:
            continue
        result = item["result"]
        fields = doc_fields(result)
        terms = matching_terms(query, fields)
        signals = item["signals"]
        output.append(
            {
                "rank": len(output) + 1,
                "score": item["score"],
                "relevance": item["relevance"],
                "signals": signals,
                "reason": result_reason(signals, terms),
                "match_terms": terms,
                "raw_scores": {
                    "semantic": item["semantic_score"],
                    "fts": item["fts_score"],
                },
                "id": doc_id(result),
                "title": fields.get("title", ""),
                "path": fields.get("path", ""),
                "chunk": fields.get("chunk", ""),
                "heading": fields.get("heading", ""),
                "type": fields.get("type", ""),
                "tags": fields.get("tags", []),
                "project": fields.get("project", ""),
                "timestamp": fields.get("timestamp", ""),
                "text": fields.get("text", ""),
            }
        )
        if len(output) >= topk:
            break
    if use_cache:
        with _QUERY_CACHE_LOCK:
            _QUERY_CACHE[cache_key] = output
            _QUERY_CACHE.move_to_end(cache_key)
            while len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
                _QUERY_CACHE.popitem(last=False)
    return output


def format_search_results(results: list[dict[str, Any]], snippet: int) -> str:
    lines: list[str] = []
    for item in results:
        text = str(item.get("text", "")).replace("\n", " ")
        text_snippet = text[:snippet].strip()
        lines.append(
            f"{item['rank']}. relevance={item['relevance']:.4f} "
            f"score={item['score']:.4f} id={item['id']}"
        )
        lines.append(f"   title: {item.get('title', '')}")
        lines.append(f"   path:  {item.get('path', '')}")
        if item.get("heading"):
            lines.append(f"   heading: {item.get('heading', '')}")
        lines.append(f"   reason: {item.get('reason', '')}")
        if text_snippet:
            lines.append(f"   text:  {text_snippet}")
    return "\n".join(lines)


def read_token_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def service_token() -> str:
    return read_token_file(_SERVICE_TOKEN_FILE)


def search_token() -> str:
    return read_token_file(_SEARCH_TOKEN_FILE)


def admin_token() -> str:
    return read_token_file(_ADMIN_TOKEN_FILE)


def is_authorized(headers: Any) -> bool:
    expected = service_token()
    if not expected:
        return False
    return secrets.compare_digest(headers.get("X-OKF-Zvec-Token", ""), expected)


def basic_password(authorization: str) -> str:
    if not authorization.startswith("Basic "):
        return ""
    try:
        decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""
    username, separator, password = decoded.partition(":")
    return password if separator and username == "okf" else ""


def is_search_authorized(headers: Any) -> bool:
    expected = search_token()
    if not expected:
        return os.environ.get("OKF_ZVEC_ALLOW_ANONYMOUS_SEARCH", "").casefold() in (
            "1",
            "true",
            "yes",
        )
    candidates = (
        headers.get("X-OKF-Zvec-Search-Token", ""),
        headers.get("Authorization", "").removeprefix("Bearer ").strip(),
        basic_password(headers.get("Authorization", "")),
    )
    return any(candidate and secrets.compare_digest(candidate, expected) for candidate in candidates)


def is_admin_authorized(headers: Any) -> bool:
    expected = admin_token()
    if not expected:
        return False
    candidates = (
        headers.get("X-OKF-Zvec-Admin-Token", ""),
        headers.get("Authorization", "").removeprefix("Bearer ").strip(),
    )
    return any(candidate and secrets.compare_digest(candidate, expected) for candidate in candidates)


def is_ai_search(headers: Any) -> bool:
    return headers.get("X-OKF-Zvec-Origin", "").strip().casefold() == "ai"


def load_ai_history_locked() -> None:
    global _AI_HISTORY_LOADED

    if _AI_HISTORY_LOADED:
        return
    try:
        payload = json.loads(_AI_HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            _AI_HISTORY.extend(item for item in payload[-_AI_HISTORY_LIMIT:] if isinstance(item, dict))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    _AI_HISTORY_LOADED = True


def record_ai_search(entry: dict[str, Any]) -> None:
    with _AI_HISTORY_LOCK:
        load_ai_history_locked()
        _AI_HISTORY.append(entry)
        try:
            _AI_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = _AI_HISTORY_FILE.with_suffix(_AI_HISTORY_FILE.suffix + ".tmp")
            temporary.write_text(
                json.dumps(list(_AI_HISTORY), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(_AI_HISTORY_FILE)
        except OSError as exc:
            log_event("ai_history_write_failed", level="error", error_type=type(exc).__name__)


def ai_history_snapshot() -> list[dict[str, Any]]:
    with _AI_HISTORY_LOCK:
        load_ai_history_locked()
        return list(reversed(_AI_HISTORY))


def collection_doc_count(collection: Any) -> int:
    try:
        stats = json.loads(str(collection.stats))
        return int(stats.get("doc_count", 0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0


def status_snapshot() -> dict[str, Any]:
    with _STATE_LOCK:
        state = json.loads(json.dumps(_SERVICE_STATE, default=str))
    with _QUERY_CACHE_LOCK:
        cache_entries = len(_QUERY_CACHE)
    with _MODEL_LOCK:
        loaded_models = sorted(_MODELS)
    preload_setting, _ = configured_preload_setting()
    state.update({
        "uptime_seconds": round(time.time() - SERVICE_STARTED_AT, 1),
        "cache_entries": cache_entries,
        "cache_limit": _QUERY_CACHE_MAX,
        "loaded_models": loaded_models,
        "retained_versions": keep_versions(),
        "search_auth_enabled": bool(search_token()),
        "preload_models": preload_setting,
        "indexed_models": list(_SEARCH_COLLECTIONS),
    })
    return state


def prometheus_metrics() -> str:
    with _QUERY_CACHE_LOCK:
        cache_entries = len(_QUERY_CACHE)
    lines = [
        "# HELP okf_zvec_uptime_seconds Время работы сервиса.",
        "# TYPE okf_zvec_uptime_seconds gauge",
        f"okf_zvec_uptime_seconds {time.time() - SERVICE_STARTED_AT:.3f}",
        "# HELP okf_zvec_cache_entries Число ответов в кэше.",
        "# TYPE okf_zvec_cache_entries gauge",
        f"okf_zvec_cache_entries {cache_entries}",
    ]
    with _METRICS.lock:
        lines.extend([
            "# HELP okf_zvec_cache_requests_total Обращения к кэшу поиска.",
            "# TYPE okf_zvec_cache_requests_total counter",
            f'okf_zvec_cache_requests_total{{result="hit"}} {_METRICS.cache_hits}',
            f'okf_zvec_cache_requests_total{{result="miss"}} {_METRICS.cache_misses}',
            "# HELP okf_zvec_search_requests_total Поисковые запросы.",
            "# TYPE okf_zvec_search_requests_total counter",
        ])
        for (model, mode, status), value in sorted(_METRICS.search_requests.items()):
            lines.append(
                f'okf_zvec_search_requests_total{{model="{model}",mode="{mode}",'
                f'status="{status}"}} {value}'
            )
        lines.extend([
            "# HELP okf_zvec_search_duration_seconds Время выполнения поиска.",
            "# TYPE okf_zvec_search_duration_seconds summary",
        ])
        for (model, mode), value in sorted(_METRICS.search_duration_sum.items()):
            labels = f'model="{model}",mode="{mode}"'
            lines.append(f"okf_zvec_search_duration_seconds_sum{{{labels}}} {value:.6f}")
            lines.append(
                f"okf_zvec_search_duration_seconds_count{{{labels}}} "
                f"{_METRICS.search_duration_count[(model, mode)]}"
            )
        lines.extend([
            "# HELP okf_zvec_sync_total Синхронизации OKF.",
            "# TYPE okf_zvec_sync_total counter",
        ])
        for status, value in sorted(_METRICS.sync_total.items()):
            lines.append(f'okf_zvec_sync_total{{status="{status}"}} {value}')
        lines.extend([
            "# HELP okf_zvec_sync_duration_seconds Время синхронизации.",
            "# TYPE okf_zvec_sync_duration_seconds summary",
            f"okf_zvec_sync_duration_seconds_sum {_METRICS.sync_duration_sum:.6f}",
            f"okf_zvec_sync_duration_seconds_count {_METRICS.sync_duration_count}",
            "# HELP okf_zvec_model_loaded Загружена ли модель в память.",
            "# TYPE okf_zvec_model_loaded gauge",
        ])
        with _MODEL_LOCK:
            loaded_models = set(_MODELS)
        for model_key in MODEL_CONFIGS:
            loaded = 1 if model_key in loaded_models else 0
            lines.append(f'okf_zvec_model_loaded{{model="{model_key}"}} {loaded}')
        lines.extend([
            "# HELP okf_zvec_model_load_seconds Время загрузки модели.",
            "# TYPE okf_zvec_model_load_seconds gauge",
        ])
        for model, duration in sorted(_METRICS.model_load_seconds.items()):
            lines.append(f'okf_zvec_model_load_seconds{{model="{model}"}} {duration:.6f}')
    with _STATE_LOCK:
        lines.extend([
            "# HELP okf_zvec_index_documents Число фрагментов в индексе.",
            "# TYPE okf_zvec_index_documents gauge",
        ])
        for model, model_state in sorted(_SERVICE_STATE["models"].items()):
            lines.append(
                f'okf_zvec_index_documents{{model="{model}"}} '
                f'{int(model_state.get("doc_count", 0))}'
            )
    return "\n".join(lines) + "\n"


def safe_extract_tar(archive: Path, target: Path) -> None:
    target_root = target.resolve()
    max_members = env_int("OKF_ZVEC_MAX_ARCHIVE_MEMBERS", DEFAULT_MAX_ARCHIVE_MEMBERS)
    max_extracted = env_int("OKF_ZVEC_MAX_EXTRACTED_BYTES", DEFAULT_MAX_EXTRACTED_BYTES)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > max_members:
            raise RequestError(413, f"в архиве больше {max_members} элементов")
        total_size = 0
        for member in members:
            member_target = (target_root / member.name).resolve()
            if not member_target.is_relative_to(target_root):
                raise RequestError(400, "архив содержит небезопасный путь")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise RequestError(400, "архив содержит неподдерживаемый тип элемента")
            if member.isfile():
                total_size += member.size
                if total_size > max_extracted:
                    raise RequestError(413, "распакованный архив превышает допустимый размер")
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError("требуется Python с поддержкой безопасных фильтров tarfile")
        tar.extractall(target_root, members=members, filter="data")


def model_db_dir(db_root: Path, model_key: str) -> Path:
    return db_root / normalize_model_key(model_key)


def build_index(okf_dir: Path, db_dir: Path, model_key: str = DEFAULT_MODEL) -> tuple[Any, int]:
    model_key = normalize_model_key(model_key)
    if db_dir.exists():
        shutil.rmtree(db_dir)

    collection = zvec.create_and_open(str(db_dir), create_schema())
    batch_size = env_int("OKF_ZVEC_INDEX_BATCH_SIZE", DEFAULT_INDEX_BATCH_SIZE)
    batch: list[dict[str, Any]] = []
    doc_count = 0
    for item in iter_doc_items(okf_dir):
        batch.append(item)
        if len(batch) >= batch_size:
            docs = docs_from_items(batch, model_key)
            collection.insert(docs)
            doc_count += len(docs)
            batch.clear()
    if batch:
        docs = docs_from_items(batch, model_key)
        collection.insert(docs)
        doc_count += len(docs)
    if doc_count:
        collection.optimize()
        collection.flush()
    return collection, doc_count


def read_active_db_root(default_db_root: Path) -> Path:
    candidates = (
        _ACTIVE_DB_FILE,
        default_db_root.parent / "active-db-root",
        default_db_root.parent / ".active-db-root",
    )
    for active_file in candidates:
        try:
            value = active_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if value:
            return Path(value).resolve()
    return default_db_root


def write_active_db_root(db_root: Path) -> None:
    _ACTIVE_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = _ACTIVE_DB_FILE.with_name(f".{_ACTIVE_DB_FILE.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(str(db_root.resolve()), encoding="utf-8")
    os.replace(temporary, _ACTIVE_DB_FILE)


def keep_versions() -> int:
    raw_value = os.environ.get("OKF_ZVEC_KEEP_VERSIONS", str(DEFAULT_KEEP_VERSIONS))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("OKF_ZVEC_KEEP_VERSIONS должно быть целым числом") from exc
    if value < 1:
        raise ValueError("OKF_ZVEC_KEEP_VERSIONS должно быть не меньше 1")
    return value


def cleanup_db_versions(db_root: Path, active_db_root: Path, keep: int) -> dict[str, list[str]]:
    candidates = [
        path
        for path in db_root.parent.glob(f"{db_root.name}-*")
        if path.is_dir()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)

    retained = {active_db_root.resolve()}
    for path in candidates:
        if len(retained) >= keep:
            break
        retained.add(path.resolve())

    deleted: list[str] = []
    skipped: list[str] = []
    for path in candidates:
        if path.resolve() in retained:
            continue
        try:
            shutil.rmtree(path)
            deleted.append(str(path))
        except OSError:
            skipped.append(str(path))
    return {"deleted": deleted, "skipped": skipped}


def stage_okf_directory(uploaded_okf: Path, okf_dir: Path, version: str) -> Path:
    staged = okf_dir.parent / f".{okf_dir.name}.staging-{version}"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(uploaded_okf, staged)
    return staged


def activate_staged_okf(staged_okf: Path, okf_dir: Path, version: str) -> Path | None:
    backup = okf_dir.parent / f".{okf_dir.name}.backup-{version}"
    if backup.exists():
        shutil.rmtree(backup)
    if okf_dir.exists():
        os.replace(okf_dir, backup)
    else:
        backup = None
    try:
        os.replace(staged_okf, okf_dir)
    except Exception:
        if backup is not None:
            os.replace(backup, okf_dir)
        raise
    return backup


def rollback_okf_activation(okf_dir: Path, backup: Path | None) -> None:
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(okf_dir)
    if backup is not None and backup.exists():
        os.replace(backup, okf_dir)


def sync_okf_from_archive(archive: Path, okf_dir: Path, db_root: Path) -> dict[str, Any]:
    global _SEARCH_COLLECTIONS

    if not _SYNC_LOCK.acquire(blocking=False):
        raise RequestError(409, "синхронизация уже выполняется")
    if _RESTART_REQUESTED.is_set():
        _SYNC_LOCK.release()
        raise RequestError(503, "сервис перезапускается")
    _SYNC_IN_PROGRESS.set()
    try:
        version = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        staged_okf: Path | None = None
        next_db_root = db_root.parent / f"{db_root.name}-{version}"
        built: dict[str, Any] = {}
        response_models: dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="okf-sync-") as tmp:
            tmp_path = Path(tmp)
            safe_extract_tar(archive, tmp_path)
            uploaded_okf = tmp_path / "okf"
            if not uploaded_okf.is_dir():
                raise ValueError("загруженный архив должен содержать каталог okf")
            staged_okf = stage_okf_directory(uploaded_okf, okf_dir, version)

        try:
            for model_key in configured_index_models():
                collection, doc_count = build_index(
                    staged_okf,
                    model_db_dir(next_db_root, model_key),
                    model_key,
                )
                built[model_key] = collection
                response_models[model_key] = {
                    "doc_count": doc_count,
                    "stats": str(collection.stats),
                    "name": MODEL_CONFIGS[model_key]["name"],
                }
        except Exception:
            built.clear()
            gc.collect()
            shutil.rmtree(next_db_root, ignore_errors=True)
            shutil.rmtree(staged_okf, ignore_errors=True)
            raise

        with _SEARCH_LOCK:
            old_active_db_root = read_active_db_root(db_root)
            backup_okf: Path | None = None
            try:
                backup_okf = activate_staged_okf(staged_okf, okf_dir, version)
                write_active_db_root(next_db_root)
            except Exception:
                if backup_okf is not None:
                    rollback_okf_activation(okf_dir, backup_okf)
                built.clear()
                gc.collect()
                shutil.rmtree(next_db_root, ignore_errors=True)
                shutil.rmtree(staged_okf, ignore_errors=True)
                with contextlib.suppress(Exception):
                    write_active_db_root(old_active_db_root)
                raise

            _SEARCH_COLLECTIONS = built
            with _MODEL_LOCK:
                for model_key in list(_MODELS):
                    if model_key not in built:
                        _MODELS.pop(model_key, None)
            clear_query_cache()
            gc.collect()
        if backup_okf is not None:
            shutil.rmtree(backup_okf, ignore_errors=True)

        cleanup = cleanup_db_versions(db_root, next_db_root, keep_versions())
        return {
            "ok": True,
            "active_db_root": str(next_db_root),
            "models": response_models,
            "cleanup": cleanup,
        }
    finally:
        _SYNC_IN_PROGRESS.clear()
        _SYNC_LOCK.release()


def command_index(args: argparse.Namespace) -> None:
    okf_dir = Path(args.okf).resolve()
    db_root = Path(args.db).resolve()
    model_keys = list(MODEL_CONFIGS) if args.model == "all" else [normalize_model_key(args.model)]
    for model_key in model_keys:
        db_dir = model_db_dir(db_root, model_key)
        collection, doc_count = build_index(okf_dir, db_dir, model_key)
        print(f"Проиндексировано фрагментов OKF Markdown: {doc_count}; каталог: {db_dir}; модель: {model_key}")
        print(collection.stats)


def command_search(args: argparse.Namespace) -> None:
    model_key = normalize_model_key(args.model)
    db_root = Path(args.db).resolve()
    db_dir = model_db_dir(read_active_db_root(db_root), model_key)
    collection = zvec.open(str(db_dir))
    query = args.query
    if args.query_b64:
        query = base64.b64decode(args.query_b64).decode("utf-8")
    if not query:
        raise SystemExit("укажите поисковый запрос или --query-b64")

    options = make_search_options(
        semantic_weight=args.semantic_weight,
        fts_weight=args.fts_weight,
        min_relevance=args.min_relevance,
        doc_type=args.type,
        tags=args.tags,
        path_pattern=args.path,
        project=args.project,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    results = search_collection(
        collection,
        query,
        args.topk,
        args.rerank_pool,
        model_key,
        args.mode,
        options,
    )
    print(format_search_results(results, args.snippet))


def benchmark_rank(results: list[dict[str, Any]], expected: str) -> int:
    needle = expected.casefold()
    for item in results:
        haystack = "\n".join(
            str(item.get(name, "")) for name in ("title", "heading", "path", "text")
        ).casefold()
        if needle in haystack:
            return int(item["rank"])
    return 0


def benchmark_ranks(results: list[dict[str, Any]], relevant: list[str]) -> list[int]:
    return [benchmark_rank(results, marker) for marker in relevant]


def benchmark_ranking_metrics(ranks: list[int], topk: int) -> tuple[float, float]:
    if not ranks:
        return 0.0, 0.0
    found_ranks = sorted({rank for rank in ranks if 0 < rank <= topk})
    recall = sum(0 < rank <= topk for rank in ranks) / len(ranks)
    dcg = sum(1 / math.log2(rank + 1) for rank in found_ranks)
    ideal_count = min(len(ranks), topk)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return recall, dcg / ideal_dcg if ideal_dcg else 0.0


def benchmark_http_search(
    service_url: str,
    query: str,
    topk: int,
    rerank_pool: int,
    model_key: str,
    mode: str,
    options: SearchOptions,
    token: str = "",
) -> list[dict[str, Any]]:
    params = urlencode({
        "q": query,
        "topk": topk,
        "rerank_pool": rerank_pool,
        "model": model_key,
        "mode": mode,
        "semantic_weight": options.semantic_weight,
        "fts_weight": options.fts_weight,
        "min_relevance": options.min_relevance,
        "no_cache": 1,
    })
    request = Request(f"{service_url.rstrip('/')}/search?{params}")
    if token:
        credentials = base64.b64encode(f"okf:{token}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {credentials}")
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["results"]


def command_benchmark(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.file).resolve()
    tests = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(tests, list) or not tests:
        raise SystemExit("benchmark-файл должен содержать непустой JSON-массив")

    model_key = normalize_model_key(args.model)
    collection = None
    if not args.service_url:
        db_root = Path(args.db).resolve()
        db_dir = model_db_dir(read_active_db_root(db_root), model_key)
        collection = zvec.open(str(db_dir))
    modes = [normalize_search_mode(mode) for mode in args.modes.split(",") if mode.strip()]
    options = make_search_options(
        semantic_weight=args.semantic_weight,
        fts_weight=args.fts_weight,
        min_relevance=args.min_relevance,
    )
    benchmark_token = ""
    if args.token_file:
        benchmark_token = Path(args.token_file).read_text(encoding="utf-8").strip()
    report: dict[str, Any] = {
        "model": model_key,
        "queries": len(tests),
        "topk": args.topk,
        "modes": {},
    }

    if not args.service_url and any(mode in ("semantic", "hybrid") for mode in modes):
        with contextlib.redirect_stdout(sys.stderr):
            get_model(model_key)

    for mode in modes:
        rows: list[dict[str, Any]] = []
        for test in tests:
            query = str(test["query"])
            raw_relevant = test.get("relevant_contains")
            if raw_relevant is None:
                raw_relevant = [test["expected_contains"]]
            if not isinstance(raw_relevant, list) or not raw_relevant:
                raise SystemExit("relevant_contains должен быть непустым JSON-массивом")
            relevant = [str(marker) for marker in raw_relevant]
            clear_query_cache()
            started = time.perf_counter()
            if args.service_url:
                results = benchmark_http_search(
                    args.service_url,
                    query,
                    args.topk,
                    args.rerank_pool,
                    model_key,
                    mode,
                    options,
                    benchmark_token,
                )
            else:
                results = search_collection(
                    collection,
                    query,
                    args.topk,
                    args.rerank_pool,
                    model_key,
                    mode,
                    options,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            ranks = benchmark_ranks(results, relevant)
            rank = min((rank for rank in ranks if rank), default=0)
            recall_at_k, ndcg_at_k = benchmark_ranking_metrics(ranks, args.topk)
            rows.append({
                "query": query,
                "relevant_contains": relevant,
                "rank": rank,
                "relevant_ranks": ranks,
                "recall_at_k": recall_at_k,
                "ndcg_at_k": ndcg_at_k,
                "elapsed_ms": round(elapsed_ms, 2),
            })

        reciprocal_ranks = [1 / row["rank"] if row["rank"] else 0 for row in rows]
        report["modes"][mode] = {
            "top1": sum(row["rank"] == 1 for row in rows) / len(rows),
            "top3": sum(0 < row["rank"] <= 3 for row in rows) / len(rows),
            "mrr": sum(reciprocal_ranks) / len(rows),
            "recall_at_k": sum(row["recall_at_k"] for row in rows) / len(rows),
            "ndcg_at_k": sum(row["ndcg_at_k"] for row in rows) / len(rows),
            "no_result": sum(row["rank"] == 0 for row in rows),
            "average_ms": sum(row["elapsed_ms"] for row in rows) / len(rows),
            "results": rows,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


class SearchHandler(BaseHTTPRequestHandler):
    server_version = "OkfZvecSearch/0.6.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(
        self,
        status: int,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def send_html(
        self,
        status: int,
        page: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, status: int, body_text: str, content_type: str) -> None:
        body = body_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def send_request_error(self, exc: RequestError, request_id: str | None = None) -> None:
        payload = {"error": exc.message}
        headers = None
        if request_id:
            payload["request_id"] = request_id
            headers = {"X-Request-Id": request_id}
        self.send_json(exc.status, payload, headers)

    def send_internal_error(self, request_id: str | None = None) -> None:
        payload = {"error": "внутренняя ошибка сервиса"}
        headers = None
        if request_id:
            payload["request_id"] = request_id
            headers = {"X-Request-Id": request_id}
        self.send_json(500, payload, headers)

    def send_search_unauthorized(self, wants_json: bool = False) -> None:
        headers = {"WWW-Authenticate": 'Basic realm="OKF Zvec Search", charset="UTF-8"'}
        if wants_json:
            self.send_json(401, {"error": "требуется авторизация"}, headers)
        else:
            self.send_html(401, "<h1>Требуется авторизация</h1>", headers)

    def send_admin_unauthorized(self) -> None:
        self.send_json(401, {"error": "требуется токен администратора"})

    def record_ai_request(
        self,
        request_id: str,
        query: str,
        model: str,
        mode: str,
        duration: float,
        status: str,
        results: list[dict[str, Any]] | None = None,
        error: str = "",
    ) -> None:
        if not is_ai_search(self.headers):
            return
        results = results or []
        top_result = results[0] if results else {}
        record_ai_search({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "query": query[:500],
            "model": model,
            "mode": mode,
            "duration_ms": round(duration * 1000, 1),
            "status": status,
            "result_count": len(results),
            "top_title": str(top_result.get("title", "")),
            "top_path": str(top_result.get("path", "")),
            "top_relevance": top_result.get("relevance"),
            "error": error,
        })

    def render_home(self) -> str:
        model_options = "".join(
            f'<option value="{model_key}">{html.escape(MODEL_CONFIGS[model_key]["name"])}</option>'
            for model_key in _SEARCH_COLLECTIONS
        )
        page = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Поиск OKF zvec</title>
  <style>
    body { font-family: system-ui, Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }
    main { max-width: 980px; margin: 0 auto; }
    form { margin: 20px 0; }
    .search-row, .filters { display: flex; gap: 8px; flex-wrap: wrap; }
    .filters { margin-top: 10px; }
    input, select, button { font: inherit; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 6px; }
    #query { flex: 1; min-width: 260px; }
    .filters input { flex: 1; min-width: 140px; }
    input[type=number] { width: 76px; }
    button { background: #0f766e; color: white; border-color: #0f766e; cursor: pointer; }
    details { margin-top: 10px; color: #475569; }
    summary { cursor: pointer; }
    article { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; margin: 10px 0; }
    .meta { color: #64748b; font-size: 13px; margin-bottom: 6px; }
    .score { font-variant-numeric: tabular-nums; }
    .reason { color: #475569; font-size: 14px; margin-top: 6px; }
    mark { background: #fde68a; color: inherit; padding: 0 1px; }
    pre { white-space: pre-wrap; margin: 8px 0 0; font: inherit; }
  </style>
</head>
<body>
<main>
  <div class="search-row">
    <h1 style="flex:1">Поиск по базе OKF</h1>
    <span><a href="/ai-history">Запросы ИИ</a> · <a href="/status">Состояние</a></span>
  </div>
  <form id="searchForm">
    <div class="search-row">
      <input id="query" type="text" placeholder="миграция, портал поставщиков, прошивка кассы" autofocus>
      <select id="model">
        __MODEL_OPTIONS__
      </select>
      <select id="mode">
        <option value="hybrid">Семантика + FTS</option>
        <option value="semantic">Только семантика</option>
        <option value="fts">Только FTS</option>
        <option value="fts_raw">FTS без лемматизации</option>
      </select>
      <input id="topk" type="number" min="1" max="20" value="5" title="Количество результатов">
      <button type="submit">Найти</button>
    </div>
    <details>
      <summary>Фильтры и ранжирование</summary>
      <div class="filters">
        <input id="filterType" type="text" placeholder="Тип">
        <input id="filterTags" type="text" placeholder="Теги через запятую">
        <input id="filterPath" type="text" placeholder="Путь: topics/*">
        <input id="filterProject" type="text" placeholder="Проект">
        <input id="dateFrom" type="text" placeholder="Дата от">
        <input id="dateTo" type="text" placeholder="Дата до">
        <input id="minRelevance" type="number" min="0" max="1" step="0.05" value="0.25" title="Минимальная релевантность">
        <input id="semanticWeight" type="number" min="0" step="0.1" value="1" title="Вес семантики">
        <input id="ftsWeight" type="number" min="0" step="0.1" value="1" title="Вес FTS">
      </div>
    </details>
  </form>
  <div id="status"></div>
  <section id="results"></section>
</main>
<script>
const form = document.getElementById('searchForm');
const query = document.getElementById('query');
const topk = document.getElementById('topk');
const model = document.getElementById('model');
const mode = document.getElementById('mode');
const filterType = document.getElementById('filterType');
const filterTags = document.getElementById('filterTags');
const filterPath = document.getElementById('filterPath');
const filterProject = document.getElementById('filterProject');
const dateFrom = document.getElementById('dateFrom');
const dateTo = document.getElementById('dateTo');
const minRelevance = document.getElementById('minRelevance');
const semanticWeight = document.getElementById('semanticWeight');
const ftsWeight = document.getElementById('ftsWeight');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const q = query.value.trim();
  if (!q) return;
  statusEl.textContent = 'Поиск...';
  resultsEl.innerHTML = '';
  const started = performance.now();
  const params = new URLSearchParams({
    q,
    topk: topk.value,
    model: model.value,
    mode: mode.value,
    type: filterType.value,
    tags: filterTags.value,
    path: filterPath.value,
    project: filterProject.value,
    date_from: dateFrom.value,
    date_to: dateTo.value,
    min_relevance: minRelevance.value,
    semantic_weight: semanticWeight.value,
    fts_weight: ftsWeight.value,
  });
  const response = await fetch(`/search?${params}`);
  const data = await response.json();
  if (!response.ok) {
    statusEl.textContent = data.error || 'Ошибка поиска';
    return;
  }
  const elapsed = Math.round(performance.now() - started);
  statusEl.textContent = `${data.model} / ${data.mode}: результатов ${data.results.length}, ${elapsed} мс`;
  resultsEl.innerHTML = data.results.map((item) => `
    <article>
      <div class="meta"><span class="score">${Math.round(Number(item.relevance) * 100)}%</span> | ${escapeHtml(item.path)}${item.heading ? ' | ' + escapeHtml(item.heading) : ''}</div>
      <strong>${escapeHtml(item.title || '')}</strong>
      <pre>${highlightText(item.text || '', item.match_terms || [])}</pre>
      <div class="reason">${escapeHtml(item.reason || '')}</div>
    </article>
  `).join('');
});
function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function highlightText(value, terms) {
  let escaped = escapeHtml(value);
  const ordered = [...terms].sort((a, b) => b.length - a.length);
  for (const term of ordered) {
    const safeTerm = escapeHtml(term);
    escaped = escaped.replace(new RegExp(escapeRegExp(safeTerm), 'giu'), '<mark>$&</mark>');
  }
  return escaped;
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
</script>
</body>
</html>"""
        return page.replace("__MODEL_OPTIONS__", model_options)

    def render_ai_history(self) -> str:
        history = ai_history_snapshot()
        history_rows = []
        for entry in history:
            timestamp = str(entry.get("timestamp", ""))
            try:
                timestamp = datetime.fromisoformat(timestamp).astimezone().strftime("%d.%m %H:%M:%S")
            except ValueError:
                pass
            top_title = str(entry.get("top_title", ""))
            top_path = str(entry.get("top_path", ""))
            top_result = top_title or top_path or "—"
            if top_title and top_path:
                top_result += f" ({top_path})"
            relevance = entry.get("top_relevance")
            relevance_text = f"{float(relevance) * 100:.0f}%" if relevance is not None else "—"
            status = "успех"
            if entry.get("status") != "success":
                status = "ошибка: " + str(entry.get("error", ""))
            history_rows.append(
                "<tr>"
                f"<td>{html.escape(timestamp)}</td>"
                f"<td class=\"history-query\">{html.escape(str(entry.get('query', '')))}</td>"
                f"<td>{html.escape(str(entry.get('model', '')))} / "
                f"{html.escape(str(entry.get('mode', '')))}</td>"
                f"<td>{float(entry.get('duration_ms', 0)):.0f} мс</td>"
                f"<td>{int(entry.get('result_count', 0))}</td>"
                f"<td>{html.escape(top_result)}</td>"
                f"<td>{relevance_text}</td>"
                f"<td>{html.escape(status)}</td>"
                "</tr>"
            )
        if not history_rows:
            history_rows.append('<tr><td colspan="8">Запросов от ИИ пока нет.</td></tr>')

        successful = [entry for entry in history if entry.get("status") == "success"]
        errors = len(history) - len(successful)
        empty = sum(int(entry.get("result_count", 0)) == 0 for entry in successful)
        average_duration = (
            sum(float(entry.get("duration_ms", 0)) for entry in successful) / len(successful)
            if successful
            else 0
        )
        relevances = [
            float(entry["top_relevance"])
            for entry in successful
            if entry.get("top_relevance") is not None
        ]
        average_relevance = sum(relevances) / len(relevances) * 100 if relevances else 0
        return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Запросы ИИ · OKF zvec</title>
  <style>
    body {{ font-family: system-ui, Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }}
    main {{ max-width: 1240px; margin: 0 auto; }}
    .summary {{ color: #475569; margin-bottom: 20px; }}
    .table-scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ text-align: left; padding: 10px; border: 1px solid #e2e8f0; vertical-align: top; }}
    th {{ white-space: nowrap; }}
    .history-query {{ min-width: 190px; max-width: 300px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
<main>
  <p><a href="/">Поиск</a> · <a href="/status">Состояние</a></p>
  <h1>Последние запросы ИИ</h1>
  <p class="summary">В выборке: {len(history)} из {_AI_HISTORY_LIMIT} · без результатов: {empty} · ошибок: {errors} · среднее время: {average_duration:.0f} мс · средняя релевантность первого результата: {average_relevance:.0f}%</p>
  <div class="table-scroll">
    <table>
      <thead><tr><th>Время</th><th>Запрос</th><th>Модель / режим</th><th>Длительность</th><th>Найдено</th><th>Лучший результат</th><th>Релевантность</th><th>Статус</th></tr></thead>
      <tbody>{''.join(history_rows)}</tbody>
    </table>
  </div>
</main>
</body>
</html>"""

    def render_status(self) -> str:
        state = status_snapshot()
        _, preload_models = configured_preload_setting()
        model_rows = []
        for model_key, config in MODEL_CONFIGS.items():
            model_state = state["models"].get(model_key, {})
            is_loaded = model_key in state["loaded_models"]
            is_indexed = model_key in _SEARCH_COLLECTIONS
            if not is_indexed:
                model_actions = "индекс отключён"
            elif is_loaded:
                model_actions = (
                    f'<a href="#" data-action="model-unload" data-model="{model_key}">Выгрузить</a> '
                    f'<a href="#" data-action="model-reload" data-model="{model_key}">Перезагрузить</a>'
                )
            else:
                model_actions = (
                    f'<a href="#" data-action="model-load" data-model="{model_key}">Загрузить</a>'
                )
            model_rows.append(
                "<tr>"
                f"<td>{html.escape(model_key)}</td>"
                f"<td>{html.escape(config['name'])}</td>"
                f"<td>{'нет индекса' if not is_indexed else ('загружена' if is_loaded else 'не загружена')}</td>"
                f"<td>{int(model_state.get('doc_count', 0))}</td>"
                f"<td class=\"model-row-actions\">{model_actions}</td>"
                "</tr>"
            )
        preload_checkboxes = "".join(
            f'<label><input type="checkbox" name="preload_models" value="{model_key}"'
            f'{" checked" if model_key in preload_models else ""}> '
            f"{html.escape(MODEL_CONFIGS[model_key]['name'])}</label>"
            for model_key in _SEARCH_COLLECTIONS
        )
        return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Состояние OKF zvec</title>
  <style>
    body {{ font-family: system-ui, Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ text-align: left; padding: 10px; border: 1px solid #e2e8f0; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 8px 16px; }}
    dt {{ color: #64748b; }}
    .model-settings {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
    .model-settings label {{ display: inline-flex; align-items: center; gap: 6px; }}
    .model-row-actions {{ white-space: nowrap; }}
    .model-row-actions a + a {{ margin-left: 12px; }}
    .service-actions {{ margin-top: 14px; }}
    .action-status {{ color: #64748b; min-height: 24px; margin-top: 8px; }}
  </style>
</head>
<body>
<main>
  <p><a href="/">Поиск</a> · <a href="/ai-history">Запросы ИИ</a></p>
  <h1>Состояние сервиса</h1>
  <dl>
    <dt>Активный индекс</dt><dd>{html.escape(str(state['active_db_root']))}</dd>
    <dt>Последняя синхронизация</dt><dd>{html.escape(str(state['last_sync_at'] or 'нет'))}</dd>
    <dt>Статус синхронизации</dt><dd>{html.escape(str(state['last_sync_status']))}</dd>
    <dt>Длительность</dt><dd>{float(state['last_sync_duration_seconds']):.2f} с</dd>
    <dt>Версий индекса</dt><dd>{int(state['retained_versions'])}</dd>
    <dt>Записей в кэше</dt><dd>{int(state['cache_entries'])} / {int(state['cache_limit'])}</dd>
    <dt>Время работы</dt><dd>{float(state['uptime_seconds']):.1f} с</dd>
    <dt>Авторизация поиска</dt><dd>{'включена' if state['search_auth_enabled'] else 'отключена'}</dd>
  </dl>
  <h2>Модели</h2>
  <table>
    <thead><tr><th>Ключ</th><th>Модель</th><th>В памяти</th><th>Фрагментов</th><th>Действие</th></tr></thead>
    <tbody>{''.join(model_rows)}</tbody>
  </table>
  <h3>Автозагрузка после перезапуска</h3>
  <form id="preloadForm" class="model-settings">
    {preload_checkboxes}
    <a href="#" data-action="save-preload">Сохранить и применить</a>
  </form>
  <p class="service-actions"><a href="#" data-action="restart">Перезапустить сервис</a></p>
  <div id="actionStatus" class="action-status"></div>
</main>
<script>
const actionStatus = document.getElementById('actionStatus');
const preloadForm = document.getElementById('preloadForm');

async function postAction(path, body) {{
  actionStatus.textContent = 'Выполняется...';
  let adminToken = sessionStorage.getItem('okfZvecAdminToken');
  if (!adminToken) {{
    adminToken = window.prompt('Введите токен администратора') || '';
    if (!adminToken) throw new Error('токен администратора не указан');
    sessionStorage.setItem('okfZvecAdminToken', adminToken);
  }}
  const response = await fetch(path, {{
    method: 'POST',
    credentials: 'same-origin',
    headers: {{
      'X-OKF-Zvec-Action': '1',
      'X-OKF-Zvec-Admin-Token': adminToken,
      ...(body ? {{'Content-Type': 'application/x-www-form-urlencoded'}} : {{}}),
    }},
    body,
  }});
  if (!response.ok && response.status !== 202) {{
    if (response.status === 401) sessionStorage.removeItem('okfZvecAdminToken');
    const payload = await response.json().catch(() => ({{}}));
    throw new Error(payload.error || `HTTP ${{response.status}}`);
  }}
  return response;
}}

document.querySelectorAll('[data-action]').forEach((link) => {{
  link.addEventListener('click', async (event) => {{
    event.preventDefault();
    try {{
      const action = link.dataset.action;
      if (action === 'save-preload') {{
        const selected = [...preloadForm.querySelectorAll('input[name="preload_models"]:checked')]
          .map((input) => input.value);
        const body = new URLSearchParams({{preload_models: selected.join(',')}});
        await postAction('/settings', body);
        window.location.reload();
      }} else if (action === 'model-load' || action === 'model-unload' || action === 'model-reload') {{
        const operation = action.replace('model-', '');
        await postAction(`/models/${{link.dataset.model}}/${{operation}}`);
        window.location.reload();
      }} else {{
        await postAction('/actions/restart');
        actionStatus.textContent = 'Сервис перезапускается...';
        setTimeout(() => window.location.reload(), 12000);
      }}
    }} catch (error) {{
      actionStatus.textContent = `Ошибка: ${{error.message}}`;
    }}
  }});
}});
</script>
</body>
</html>"""

    def do_GET(self) -> None:
        global _SEARCH_COLLECTIONS

        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "default_model": service_default_model(),
                    "models": {
                        key: MODEL_CONFIGS[key]["name"] for key in _SEARCH_COLLECTIONS
                    },
                },
            )
            return
        if not is_search_authorized(self.headers):
            self.send_search_unauthorized(wants_json=parsed.path != "/")
            return
        if parsed.path == "/":
            self.send_html(200, self.render_home())
            return
        if parsed.path == "/status":
            self.send_html(200, self.render_status())
            return
        if parsed.path == "/ai-history":
            self.send_html(200, self.render_ai_history())
            return
        if parsed.path == "/status.json":
            self.send_json(200, status_snapshot())
            return
        if parsed.path == "/metrics":
            self.send_text(200, prometheus_metrics(), "text/plain; version=0.0.4; charset=utf-8")
            return
        if parsed.path == "/models":
            self.send_json(
                200,
                {
                    "default_model": service_default_model(),
                    "models": [
                        {"key": key, "name": MODEL_CONFIGS[key]["name"]}
                        for key in _SEARCH_COLLECTIONS
                    ],
                },
            )
            return
        if parsed.path != "/search":
            self.send_json(404, {"error": "не найдено"})
            return

        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        model_key = "unknown"
        search_mode = "unknown"
        query = ""
        try:
            params = parse_qs(parsed.query)
            query = (params.get("q") or params.get("query") or [""])[0]
            try:
                model_key = normalize_model_key(
                    (params.get("model") or [service_default_model()])[0]
                )
                search_mode = normalize_search_mode(
                    (params.get("mode") or [DEFAULT_SEARCH_MODE])[0]
                )
            except ValueError as exc:
                raise RequestError(400, str(exc)) from exc
            topk = bounded_int((params.get("topk") or ["5"])[0], "topk", 1, MAX_TOPK)
            rerank_pool = bounded_int(
                (params.get("rerank_pool") or ["50"])[0],
                "rerank_pool",
                1,
                MAX_RERANK_POOL,
            )
            snippet = bounded_int(
                (params.get("snippet") or ["220"])[0],
                "snippet",
                1,
                MAX_SNIPPET,
            )
            use_cache = (params.get("no_cache") or ["0"])[0] not in ("1", "true", "yes")
            try:
                options = make_search_options(
                    semantic_weight=(
                        float(params["semantic_weight"][0])
                        if params.get("semantic_weight")
                        else None
                    ),
                    fts_weight=(
                        float(params["fts_weight"][0]) if params.get("fts_weight") else None
                    ),
                    min_relevance=(
                        float(params["min_relevance"][0])
                        if params.get("min_relevance")
                        else None
                    ),
                    doc_type=(params.get("type") or [""])[0],
                    tags=(params.get("tags") or [""])[0],
                    path_pattern=(params.get("path") or [""])[0],
                    project=(params.get("project") or [""])[0],
                    date_from=(params.get("date_from") or [""])[0],
                    date_to=(params.get("date_to") or [""])[0],
                )
            except ValueError as exc:
                raise RequestError(400, str(exc)) from exc
            collection = _SEARCH_COLLECTIONS.get(model_key)
            if collection is None:
                raise RequestError(400, f"индекс модели {model_key} не настроен")
            results = search_collection(
                collection,
                query,
                topk,
                rerank_pool,
                model_key,
                search_mode,
                options,
                use_cache,
            )
            self.send_json(
                200,
                {
                    "query": query,
                    "model": model_key,
                    "mode": search_mode,
                    "model_name": MODEL_CONFIGS[model_key]["name"],
                    "topk": topk,
                    "min_relevance": options.min_relevance,
                    "weights": {
                        "semantic": options.semantic_weight,
                        "fts": options.fts_weight,
                    },
                    "results": [
                        {**item, "text": str(item.get("text", ""))[:snippet]} for item in results
                    ],
                },
                {"X-Request-Id": request_id},
            )
            duration = time.perf_counter() - started
            _METRICS.record_search(model_key, search_mode, "success", duration)
            self.record_ai_request(
                request_id,
                query,
                model_key,
                search_mode,
                duration,
                "success",
                results,
            )
            log_event(
                "search_completed",
                request_id=request_id,
                model=model_key,
                mode=search_mode,
                duration_seconds=round(duration, 6),
                result_count=len(results),
                cache_enabled=use_cache,
            )
        except RequestError as exc:
            duration = time.perf_counter() - started
            _METRICS.record_search(model_key, search_mode, "error", duration)
            self.record_ai_request(
                request_id, query, model_key, search_mode, duration, "error", error=exc.message
            )
            self.send_request_error(exc, request_id)
        except ModelLoadError as exc:
            duration = time.perf_counter() - started
            _METRICS.record_search(model_key, search_mode, "error", duration)
            self.record_ai_request(
                request_id,
                query,
                model_key,
                search_mode,
                duration,
                "error",
                error="модель поиска временно недоступна",
            )
            log_event(
                "search_failed",
                level="error",
                request_id=request_id,
                model=model_key,
                mode=search_mode,
                duration_seconds=round(duration, 6),
                error_type=type(exc).__name__,
            )
            self.send_json(
                503,
                {"error": "модель поиска временно недоступна", "request_id": request_id},
                {"X-Request-Id": request_id},
            )
        except Exception as exc:
            duration = time.perf_counter() - started
            _METRICS.record_search(model_key, search_mode, "error", duration)
            self.record_ai_request(
                request_id,
                query,
                model_key,
                search_mode,
                duration,
                "error",
                error="внутренняя ошибка сервиса",
            )
            log_event(
                "search_failed",
                level="error",
                request_id=request_id,
                model=model_key,
                mode=search_mode,
                duration_seconds=round(duration, 6),
                error_type=type(exc).__name__,
            )
            self.send_internal_error(request_id)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        model_action = re.fullmatch(r"/models/([^/]+)/(load|unload|reload)", parsed.path)
        if parsed.path in ("/settings", "/actions/reload-models", "/actions/restart") or model_action:
            if not is_admin_authorized(self.headers):
                self.send_admin_unauthorized()
                return
            try:
                if self.headers.get("X-OKF-Zvec-Action") != "1":
                    raise RequestError(403, "недопустимый управляющий запрос")
                if parsed.path == "/settings":
                    length = bounded_int(
                        self.headers.get("Content-Length", "0"),
                        "Content-Length",
                        1,
                        4096,
                        413,
                    )
                    body = self.rfile.read(length).decode("utf-8")
                    params = parse_qs(body)
                    value = ",".join(params.get("preload_models") or ["none"])
                    canonical, _ = apply_preload_setting(value)
                    save_preload_setting(canonical)
                    self.send_redirect("/status")
                    return
                if parsed.path == "/actions/reload-models":
                    reload_preloaded_models()
                    self.send_redirect("/status")
                    return
                if model_action:
                    model_key, operation = model_action.groups()
                    if normalize_model_key(model_key) not in _SEARCH_COLLECTIONS:
                        raise RequestError(400, f"индекс модели {model_key} не настроен")
                    if operation == "load":
                        load_model(model_key)
                    elif operation == "unload":
                        unload_model(model_key)
                    else:
                        reload_model(model_key)
                    self.send_redirect("/status")
                    return

                if not _SYNC_LOCK.acquire(blocking=False):
                    raise RequestError(409, "дождитесь завершения синхронизации")
                try:
                    if _SYNC_IN_PROGRESS.is_set():
                        raise RequestError(409, "дождитесь завершения синхронизации")
                    _RESTART_REQUESTED.set()
                finally:
                    _SYNC_LOCK.release()
                log_event("service_restart_requested")
                self.send_html(
                    202,
                    "<h1>Сервис перезапускается</h1><p>Обновите страницу через несколько секунд.</p>",
                )
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            except RequestError as exc:
                self.send_request_error(exc)
                return
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
                return
            except ModelLoadError:
                self.send_json(503, {"error": "не удалось загрузить выбранные модели"})
                return
            except Exception as exc:
                log_event(
                    "settings_action_failed",
                    level="error",
                    action=parsed.path,
                    error_type=type(exc).__name__,
                )
                self.send_internal_error()
                return

        if parsed.path != "/sync":
            self.send_json(404, {"error": "не найдено"})
            return
        if not is_authorized(self.headers):
            self.send_json(401, {"error": "нет авторизации"})
            return

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        archive_path: Path | None = None
        try:
            length = bounded_int(
                self.headers.get("Content-Length", "0"),
                "Content-Length",
                1,
                env_int("OKF_ZVEC_MAX_SYNC_BYTES", DEFAULT_MAX_SYNC_BYTES),
                413,
            )

            with tempfile.NamedTemporaryFile(prefix="okf-upload-", suffix=".tgz", delete=False) as tmp:
                archive_path = Path(tmp.name)
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    tmp.write(chunk)
                    remaining -= len(chunk)
                if remaining:
                    raise RequestError(400, "тело запроса получено не полностью")

            payload = sync_okf_from_archive(
                archive_path,
                self.server.okf_dir,
                self.server.db_dir,
            )
            duration = time.perf_counter() - started
            _METRICS.record_sync("success", duration)
            with _STATE_LOCK:
                _SERVICE_STATE.update({
                    "last_sync_at": datetime.now(timezone.utc).isoformat(),
                    "last_sync_status": "success",
                    "last_sync_duration_seconds": round(duration, 3),
                    "last_sync_error": "",
                    "active_db_root": payload["active_db_root"],
                })
                for model_key in MODEL_CONFIGS:
                    model_state = _SERVICE_STATE["models"].setdefault(model_key, {})
                    model_state["doc_count"] = 0
                for model_key, model in payload["models"].items():
                    model_state = _SERVICE_STATE["models"].setdefault(model_key, {})
                    model_state["doc_count"] = int(model["doc_count"])
            log_event(
                "sync_completed",
                request_id=request_id,
                duration_seconds=round(duration, 3),
                active_db_root=payload["active_db_root"],
                deleted_versions=len(payload["cleanup"]["deleted"]),
                skipped_versions=len(payload["cleanup"]["skipped"]),
            )
            self.send_json(200, payload, {"X-Request-Id": request_id})
        except (tarfile.ReadError, UnicodeDecodeError):
            duration = time.perf_counter() - started
            _METRICS.record_sync("error", duration)
            self.send_request_error(RequestError(400, "не удалось прочитать архив OKF"), request_id)
        except ValueError as exc:
            duration = time.perf_counter() - started
            _METRICS.record_sync("error", duration)
            self.send_request_error(RequestError(400, str(exc)), request_id)
        except RequestError as exc:
            duration = time.perf_counter() - started
            _METRICS.record_sync("error", duration)
            self.send_request_error(exc, request_id)
        except Exception as exc:
            duration = time.perf_counter() - started
            _METRICS.record_sync("error", duration)
            with _STATE_LOCK:
                _SERVICE_STATE.update({
                    "last_sync_at": datetime.now(timezone.utc).isoformat(),
                    "last_sync_status": "error",
                    "last_sync_duration_seconds": round(duration, 3),
                    "last_sync_error": "внутренняя ошибка сервиса",
                })
            log_event(
                "sync_failed",
                level="error",
                request_id=request_id,
                duration_seconds=round(duration, 3),
                error_type=type(exc).__name__,
            )
            self.send_internal_error(request_id)
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)


def command_serve(args: argparse.Namespace) -> None:
    global _SEARCH_COLLECTIONS

    db_root = Path(args.db).resolve()
    active_db_root = read_active_db_root(db_root)
    collections: dict[str, Any] = {}
    index_models = configured_index_models()
    for model_key in index_models:
        db_dir = model_db_dir(active_db_root, model_key)
        if not db_dir.exists():
            build_index(Path(args.okf).resolve(), db_dir, model_key)
        try:
            collections[model_key] = zvec.open(str(db_dir))
        except Exception:
            collection, _ = build_index(Path(args.okf).resolve(), db_dir, model_key)
            collections[model_key] = collection
    _SEARCH_COLLECTIONS = collections

    preload_value, preload_models = configured_preload_setting()
    unavailable_preloads = [model for model in preload_models if model not in collections]
    if unavailable_preloads:
        raise ValueError(
            "предзагрузка требует включённого индекса: " + ", ".join(unavailable_preloads)
        )
    for model_key in preload_models:
        get_model(model_key)

    with _STATE_LOCK:
        _SERVICE_STATE["active_db_root"] = str(active_db_root)
        if active_db_root.exists():
            _SERVICE_STATE["last_sync_at"] = datetime.fromtimestamp(
                active_db_root.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
            _SERVICE_STATE["last_sync_status"] = "loaded"
        for model_key, collection in collections.items():
            _SERVICE_STATE["models"][model_key] = {
                "name": MODEL_CONFIGS[model_key]["name"],
                "loaded": model_key in _MODELS,
                "doc_count": collection_doc_count(collection),
                "load_seconds": _METRICS.model_load_seconds.get(model_key, 0.0),
            }

    server = ThreadingHTTPServer((args.host, args.port), SearchHandler)
    server.okf_dir = Path(args.okf).resolve()
    server.db_dir = db_root
    log_event(
        "service_started",
        address=f"http://{args.host}:{args.port}",
        active_db_root=str(active_db_root),
        search_auth_enabled=bool(search_token()),
        preload_setting=preload_value,
        preloaded_models=preload_models,
        indexed_models=index_models,
    )
    server.serve_forever()
    server.server_close()
    if _RESTART_REQUESTED.is_set():
        raise SystemExit(75)


def add_quality_arguments(parser: argparse.ArgumentParser, include_filters: bool = True) -> None:
    parser.add_argument("--semantic-weight", type=float, help="вес семантической ветки")
    parser.add_argument("--fts-weight", type=float, help="вес полнотекстовой ветки")
    parser.add_argument("--min-relevance", type=float, help="минимальная релевантность от 0 до 1")
    if include_filters:
        parser.add_argument("--type", default="", help="фильтр по полю type")
        parser.add_argument("--tags", default="", help="обязательные теги через запятую")
        parser.add_argument("--path", default="", help="маска пути, например topics/*")
        parser.add_argument("--project", default="", help="фильтр по полю project")
        parser.add_argument("--date-from", default="", help="минимальная дата или timestamp")
        parser.add_argument("--date-to", default="", help="максимальная дата или timestamp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Индексирование и поиск по OKF Markdown с помощью zvec.")
    parser.add_argument("--db", default=str(DEFAULT_DB_ROOT), help="корневой каталог коллекций zvec")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="перестроить индекс OKF zvec")
    index_parser.add_argument("--okf", default=str(DEFAULT_OKF_DIR), help="каталог OKF Markdown")
    index_parser.add_argument("--model", default="all", choices=["all", *MODEL_CONFIGS.keys()])
    index_parser.set_defaults(func=command_index)

    search_parser = subparsers.add_parser("search", help="искать в индексе OKF zvec")
    search_parser.add_argument("query", nargs="?", help="поисковый запрос")
    search_parser.add_argument("--query-b64", help="поисковый запрос UTF-8 в кодировке base64")
    search_parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()))
    search_parser.add_argument("--mode", default=DEFAULT_SEARCH_MODE, choices=SEARCH_MODES)
    search_parser.add_argument("--topk", type=int, default=5)
    search_parser.add_argument("--rerank-pool", type=int, default=50)
    search_parser.add_argument("--snippet", type=int, default=220)
    add_quality_arguments(search_parser)
    search_parser.set_defaults(func=command_search)

    benchmark_parser = subparsers.add_parser("benchmark", help="сравнить качество режимов поиска")
    benchmark_parser.add_argument("--file", default="benchmarks/queries.json")
    benchmark_parser.add_argument("--service-url", help="HTTP-адрес запущенного сервиса")
    benchmark_parser.add_argument("--token-file", help="файл токена поискового API")
    benchmark_parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()))
    benchmark_parser.add_argument("--modes", default="semantic,fts_raw,fts,hybrid")
    benchmark_parser.add_argument("--topk", type=int, default=5)
    benchmark_parser.add_argument("--rerank-pool", type=int, default=50)
    add_quality_arguments(benchmark_parser, include_filters=False)
    benchmark_parser.set_defaults(func=command_benchmark)

    serve_parser = subparsers.add_parser("serve", help="запустить HTTP-сервис поиска")
    serve_parser.add_argument("--okf", default=str(DEFAULT_OKF_DIR), help="каталог OKF Markdown")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.set_defaults(func=command_serve)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
