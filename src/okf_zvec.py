#!/usr/bin/env python3
"""Локальный поиск по OKF на базе zvec."""

from __future__ import annotations

import argparse
import base64
import contextlib
import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import tarfile
import threading
import tempfile
import gc
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import zvec
import yaml


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
DIMENSION = 384
DEFAULT_MODEL = "e5"
DEFAULT_SEARCH_MODE = "hybrid"
SEARCH_MODES = ("semantic", "fts", "hybrid")
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
_QUERY_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_QUERY_CACHE_MAX = 256
_SERVICE_TOKEN_FILE = Path(
    os.environ.get("OKF_ZVEC_TOKEN_FILE", str(APP_HOME / "config" / "service-token"))
)
_ACTIVE_DB_FILE = Path(
    os.environ.get("OKF_ZVEC_ACTIVE_DB_FILE", str(APP_HOME / "data" / "active-db-root"))
)
_MORPH: Any | None = None
_MORPH_UNAVAILABLE = False
DEFAULT_KEEP_VERSIONS = 3
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_FTS_WEIGHT = 1.0
DEFAULT_MIN_RELEVANCE = 0.25
RRF_RANK_CONSTANT = 60


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


def get_model(model_key: str = DEFAULT_MODEL) -> Any:
    model_key = normalize_model_key(model_key)
    if model_key in _MODELS:
        return _MODELS[model_key]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Для семантического поиска требуется sentence-transformers. "
            "Установите зависимости проекта перед запуском сервиса."
        ) from exc

    try:
        model = SentenceTransformer(MODEL_CONFIGS[model_key]["name"], local_files_only=True)
    except Exception:
        model = SentenceTransformer(MODEL_CONFIGS[model_key]["name"])
    if hasattr(model, "get_embedding_dimension"):
        actual_dimension = model.get_embedding_dimension()
    else:
        actual_dimension = model.get_sentence_embedding_dimension()
    if actual_dimension != DIMENSION:
        raise SystemExit(
            f"Модель {MODEL_CONFIGS[model_key]['name']} возвращает {actual_dimension} измерений, "
            f"а схема zvec ожидает {DIMENSION}."
        )
    _MODELS[model_key] = model
    return model


def prefixed_text(text: str, model_key: str, kind: str) -> str:
    config = MODEL_CONFIGS[normalize_model_key(model_key)]
    prefix = config["query_prefix"] if kind == "query" else config["passage_prefix"]
    return f"{prefix}{text}"


def embed(text: str, model_key: str = DEFAULT_MODEL, kind: str = "query") -> list[float]:
    vector = get_model(model_key).encode(prefixed_text(text, model_key, kind), normalize_embeddings=True)
    return vector.astype("float32").tolist()


def embed_many(texts: list[str], model_key: str = DEFAULT_MODEL, kind: str = "passage") -> list[list[float]]:
    if not texts:
        return []
    model = get_model(model_key)
    vectors = model.encode(
        [prefixed_text(text, model_key, kind) for text in texts],
        normalize_embeddings=True,
        batch_size=16,
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


def iter_docs(okf_dir: Path, model_key: str = DEFAULT_MODEL) -> list[zvec.Doc]:
    items: list[dict[str, Any]] = []
    for path in sorted(okf_dir.rglob("*.md")):
        rel = path.relative_to(okf_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)
        title = metadata_text(meta.get("title")) or title_from_markdown(body, rel)
        doc_type = metadata_text(meta.get("type"))
        tags = metadata_tags(meta.get("tags"))
        project = metadata_text(meta.get("project"))
        timestamp = metadata_text(meta.get("timestamp"))
        for chunk_index, (heading, chunk) in enumerate(section_chunks(body)):
            safe_id = "doc_" + hashlib.sha1(f"{rel}#{chunk_index}".encode("utf-8")).hexdigest()
            search_text = f"{title}\n{doc_type}\n{' '.join(tags)}\n{project}\n{heading}\n{chunk}"
            items.append({
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
            })

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
                    "text": item["text"],
                    "search_text": item["fts_text"],
                },
            )
        )
    return docs


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
                "search_text",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.FtsIndexParam(tokenizer_name="whitespace", filters=["lowercase"]),
            ),
            zvec.FieldSchema(
                "text",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.FtsIndexParam(tokenizer_name="standard"),
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
    semantic = (
        semantic_relevance(item["semantic_score"])
        if item["semantic_score"] is not None
        else 0.0
    )
    fts = fts_relevance(item["fts_score"]) if item["fts_score"] is not None else 0.0
    total_weight = options.semantic_weight + options.fts_weight
    return (
        options.semantic_weight * semantic + options.fts_weight * fts
    ) / total_weight


def result_reason(signals: list[str], terms: list[str]) -> str:
    if signals == ["semantic"]:
        return "Семантическая близость запроса и фрагмента."
    if signals == ["fts"]:
        return f"Совпали термины: {', '.join(terms)}." if terms else "Полнотекстовое совпадение."
    if terms:
        return f"Семантическая близость и термины: {', '.join(terms)}."
    return "Результат объединённого семантического и полнотекстового поиска."


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
        cached = _QUERY_CACHE.get(cache_key)
        if cached is not None:
            return cached

    pool_size = max(topk, rerank_pool)
    semantic_results: list[Any] = []
    fts_results: list[Any] = []
    with _SEARCH_LOCK:
        if search_mode in ("semantic", "hybrid") and options.semantic_weight > 0:
            vector_query = zvec.Query("embedding", vector=embed(query, model_key, kind="query"))
            semantic_results = collection.query(vector_query, topk=pool_size)
        if search_mode in ("fts", "hybrid") and options.fts_weight > 0:
            fts_query = zvec.Query(
                "search_text",
                fts=zvec.Fts(match_string=normalize_fts_text(query)),
            )
            fts_results = collection.query(fts_query, topk=pool_size)

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
    elif search_mode == "fts":
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
        if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
            _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
        _QUERY_CACHE[cache_key] = output
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


def service_token() -> str:
    try:
        return _SERVICE_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def is_authorized(headers: Any) -> bool:
    expected = service_token()
    if not expected:
        return False
    return secrets.compare_digest(headers.get("X-OKF-Zvec-Token", ""), expected)


def safe_extract_tar(archive: Path, target: Path) -> None:
    target_root = target.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_target = (target_root / member.name).resolve()
            if not member_target.is_relative_to(target_root):
                raise ValueError(f"небезопасный путь в tar-архиве: {member.name}")
        tar.extractall(target_root, filter="data")


def model_db_dir(db_root: Path, model_key: str) -> Path:
    return db_root / normalize_model_key(model_key)


def build_index(okf_dir: Path, db_dir: Path, model_key: str = DEFAULT_MODEL) -> tuple[Any, int]:
    model_key = normalize_model_key(model_key)
    docs = iter_docs(okf_dir, model_key)
    if db_dir.exists():
        shutil.rmtree(db_dir)

    collection = zvec.create_and_open(str(db_dir), create_schema())
    if docs:
        collection.insert(docs)
        collection.optimize()
        collection.flush()
    return collection, len(docs)


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

    with _SEARCH_LOCK:
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
            for model_key in MODEL_CONFIGS:
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

        old_collections = _SEARCH_COLLECTIONS
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
        _QUERY_CACHE.clear()
        old_collections.clear()
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


def benchmark_http_search(
    service_url: str,
    query: str,
    topk: int,
    rerank_pool: int,
    model_key: str,
    mode: str,
    options: SearchOptions,
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
    with urlopen(f"{service_url.rstrip('/')}/search?{params}", timeout=30) as response:
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
    report: dict[str, Any] = {
        "model": model_key,
        "queries": len(tests),
        "topk": args.topk,
        "modes": {},
    }

    for mode in modes:
        rows: list[dict[str, Any]] = []
        for test in tests:
            query = str(test["query"])
            expected = str(test["expected_contains"])
            _QUERY_CACHE.clear()
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
            rank = benchmark_rank(results, expected)
            rows.append({
                "query": query,
                "expected_contains": expected,
                "rank": rank,
                "elapsed_ms": round(elapsed_ms, 2),
            })

        reciprocal_ranks = [1 / row["rank"] if row["rank"] else 0 for row in rows]
        report["modes"][mode] = {
            "top1": sum(row["rank"] == 1 for row in rows) / len(rows),
            "top3": sum(0 < row["rank"] <= 3 for row in rows) / len(rows),
            "mrr": sum(reciprocal_ranks) / len(rows),
            "no_result": sum(row["rank"] == 0 for row in rows),
            "average_ms": sum(row["elapsed_ms"] for row in rows) / len(rows),
            "results": rows,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


class SearchHandler(BaseHTTPRequestHandler):
    server_version = "OkfZvecSearch/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def render_home(self) -> str:
        return r"""<!doctype html>
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
  <h1>Поиск по базе OKF</h1>
  <form id="searchForm">
    <div class="search-row">
      <input id="query" type="text" placeholder="миграция, портал поставщиков, прошивка кассы" autofocus>
      <select id="model">
        <option value="e5">многоязычная E5 small</option>
        <option value="paraphrase">многоязычная paraphrase MiniLM</option>
      </select>
      <select id="mode">
        <option value="hybrid">Семантика + FTS</option>
        <option value="semantic">Только семантика</option>
        <option value="fts">Только FTS</option>
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

    def do_GET(self) -> None:
        global _SEARCH_COLLECTIONS

        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(200, self.render_home())
            return
        if parsed.path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "default_model": DEFAULT_MODEL,
                    "models": {key: value["name"] for key, value in MODEL_CONFIGS.items()},
                },
            )
            return
        if parsed.path == "/models":
            self.send_json(
                200,
                {
                    "default_model": DEFAULT_MODEL,
                    "models": [
                        {"key": key, "name": value["name"]} for key, value in MODEL_CONFIGS.items()
                    ],
                },
            )
            return
        if parsed.path != "/search":
            self.send_json(404, {"error": "не найдено"})
            return

        params = parse_qs(parsed.query)
        query = (params.get("q") or params.get("query") or [""])[0]
        model_key = normalize_model_key((params.get("model") or [DEFAULT_MODEL])[0])
        search_mode = normalize_search_mode((params.get("mode") or [DEFAULT_SEARCH_MODE])[0])
        topk = int((params.get("topk") or ["5"])[0])
        rerank_pool = int((params.get("rerank_pool") or ["50"])[0])
        snippet = int((params.get("snippet") or ["220"])[0])
        use_cache = (params.get("no_cache") or ["0"])[0] not in ("1", "true", "yes")

        try:
            options = make_search_options(
                semantic_weight=(
                    float(params["semantic_weight"][0]) if params.get("semantic_weight") else None
                ),
                fts_weight=float(params["fts_weight"][0]) if params.get("fts_weight") else None,
                min_relevance=(
                    float(params["min_relevance"][0]) if params.get("min_relevance") else None
                ),
                doc_type=(params.get("type") or [""])[0],
                tags=(params.get("tags") or [""])[0],
                path_pattern=(params.get("path") or [""])[0],
                project=(params.get("project") or [""])[0],
                date_from=(params.get("date_from") or [""])[0],
                date_to=(params.get("date_to") or [""])[0],
            )
            collection = _SEARCH_COLLECTIONS.get(model_key)
            if collection is None:
                raise RuntimeError(f"коллекция поиска не открыта для модели {model_key}")
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
            )
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/sync":
            self.send_json(404, {"error": "не найдено"})
            return
        if not is_authorized(self.headers):
            self.send_json(401, {"error": "нет авторизации"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("пустое тело запроса")

            with tempfile.NamedTemporaryFile(prefix="okf-upload-", suffix=".tgz", delete=False) as tmp:
                archive_path = Path(tmp.name)
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    tmp.write(chunk)
                    remaining -= len(chunk)

            try:
                payload = sync_okf_from_archive(
                    archive_path,
                    self.server.okf_dir,
                    self.server.db_dir,
                )
            finally:
                archive_path.unlink(missing_ok=True)
            self.send_json(200, payload)
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def command_serve(args: argparse.Namespace) -> None:
    global _SEARCH_COLLECTIONS

    db_root = Path(args.db).resolve()
    active_db_root = read_active_db_root(db_root)
    collections: dict[str, Any] = {}
    for model_key in MODEL_CONFIGS:
        db_dir = model_db_dir(active_db_root, model_key)
        if not db_dir.exists():
            build_index(Path(args.okf).resolve(), db_dir, model_key)
        try:
            collections[model_key] = zvec.open(str(db_dir))
        except Exception:
            collection, _ = build_index(Path(args.okf).resolve(), db_dir, model_key)
            collections[model_key] = collection
        get_model(model_key)
    _SEARCH_COLLECTIONS = collections
    server = ThreadingHTTPServer((args.host, args.port), SearchHandler)
    server.okf_dir = Path(args.okf).resolve()
    server.db_dir = db_root
    print(f"Сервис поиска OKF zvec доступен по адресу http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


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
    benchmark_parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()))
    benchmark_parser.add_argument("--modes", default="semantic,fts,hybrid")
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
