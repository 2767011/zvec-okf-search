#!/usr/bin/env python3
"""Локальный поиск по OKF на базе zvec."""

from __future__ import annotations

import argparse
import base64
import contextlib
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
from functools import lru_cache
from pathlib import Path
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import zvec


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
_QUERY_CACHE: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
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


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[match.end() :]


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
    items: list[tuple[str, str, str, str, str, str, str, str, str]] = []
    for path in sorted(okf_dir.rglob("*.md")):
        rel = path.relative_to(okf_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)
        title = meta.get("title") or title_from_markdown(body, rel)
        doc_type = meta.get("type", "")
        for chunk_index, (heading, chunk) in enumerate(section_chunks(body)):
            safe_id = "doc_" + hashlib.sha1(f"{rel}#{chunk_index}".encode("utf-8")).hexdigest()
            search_text = f"{title}\n{doc_type}\n{heading}\n{chunk}"
            items.append(
                (
                    safe_id,
                    rel,
                    str(chunk_index),
                    title,
                    doc_type,
                    heading,
                    chunk,
                    search_text,
                    normalize_fts_text(search_text),
                )
            )

    docs: list[zvec.Doc] = []
    for (safe_id, rel, chunk_index, title, doc_type, heading, chunk, search_text, fts_text), vector in zip(
        items, embed_many([item[7] for item in items], model_key, kind="passage"), strict=True
    ):
        docs.append(
            zvec.Doc(
                id=safe_id,
                vectors={"embedding": vector},
                fields={
                    "path": rel,
                    "chunk": chunk_index,
                    "heading": heading,
                    "title": title,
                    "type": doc_type,
                    "text": chunk,
                    "search_text": fts_text,
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


def lexical_bonus(query: str, fields: dict[str, Any]) -> float:
    normalized_query = query.casefold().strip()
    haystack = "\n".join(
        str(fields.get(name, "")) for name in ("title", "heading", "type", "path", "text")
    ).casefold()
    if not normalized_query or not haystack:
        return 0.0

    bonus = 0.0
    if normalized_query in haystack:
        bonus += 0.25

    query_tokens = lexical_tokens(normalized_query)
    haystack_tokens = lexical_tokens(haystack)
    if query_tokens:
        exact_matches = len(query_tokens & haystack_tokens)
        bonus += min(0.24, 0.06 * exact_matches)

        unmatched_query = query_tokens - haystack_tokens
        if unmatched_query:
            query_lemmas = {token_lemma(token) for token in unmatched_query}
            haystack_lemmas = {token_lemma(token) for token in haystack_tokens}
            lemma_matches = len(query_lemmas & haystack_lemmas)
            bonus += min(0.12, 0.03 * lemma_matches)
    return bonus


def rerank_results(results: list[Any], query: str, topk: int) -> list[tuple[float, Any]]:
    scored = []
    for result in results:
        fields = doc_fields(result)
        adjusted_score = doc_score(result) - lexical_bonus(query, fields)
        scored.append((adjusted_score, result))
    scored.sort(key=lambda item: item[0])
    return scored[:topk]


def search_collection(
    collection: Any,
    query: str,
    topk: int,
    rerank_pool: int,
    model_key: str = DEFAULT_MODEL,
    search_mode: str = DEFAULT_SEARCH_MODE,
) -> list[dict[str, Any]]:
    if not query:
        raise ValueError("для поиска нужен непустой запрос")

    model_key = normalize_model_key(model_key)
    search_mode = normalize_search_mode(search_mode)
    cache_key = (model_key, search_mode, query.casefold().strip(), topk)
    cached = _QUERY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _SEARCH_LOCK:
        if search_mode == "semantic":
            vector_query = zvec.Query("embedding", vector=embed(query, model_key, kind="query"))
            results = collection.query(vector_query, topk=topk)
        elif search_mode == "fts":
            fts_query = zvec.Query(
                "search_text",
                fts=zvec.Fts(match_string=normalize_fts_text(query)),
            )
            results = collection.query(fts_query, topk=topk)
        else:
            from zvec.extension.multi_vector_reranker import RrfReRanker

            vector_query = zvec.Query("embedding", vector=embed(query, model_key, kind="query"))
            fts_query = zvec.Query(
                "search_text",
                fts=zvec.Fts(match_string=normalize_fts_text(query)),
            )
            results = collection.query(
                [vector_query, fts_query],
                topk=topk,
                reranker=RrfReRanker(rank_constant=60),
            )

    output: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        fields = doc_fields(result)
        output.append(
            {
                "rank": rank,
                "score": doc_score(result),
                "id": doc_id(result),
                "title": fields.get("title", ""),
                "path": fields.get("path", ""),
                "chunk": fields.get("chunk", ""),
                "heading": fields.get("heading", ""),
                "type": fields.get("type", ""),
                "text": fields.get("text", ""),
            }
        )
    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
        _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
    _QUERY_CACHE[cache_key] = output
    return output


def format_search_results(results: list[dict[str, Any]], snippet: int) -> str:
    lines: list[str] = []
    for item in results:
        text = str(item.get("text", "")).replace("\n", " ")
        text_snippet = text[:snippet].strip()
        lines.append(f"{item['rank']}. score={item['score']:.4f} id={item['id']}")
        lines.append(f"   title: {item.get('title', '')}")
        lines.append(f"   path:  {item.get('path', '')}")
        if item.get("heading"):
            lines.append(f"   heading: {item.get('heading', '')}")
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
    try:
        value = _ACTIVE_DB_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default_db_root
    return Path(value).resolve() if value else default_db_root


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
    db_dir = model_db_dir(Path(args.db).resolve(), model_key)
    collection = zvec.open(str(db_dir))
    query = args.query
    if args.query_b64:
        query = base64.b64decode(args.query_b64).decode("utf-8")
    if not query:
        raise SystemExit("укажите поисковый запрос или --query-b64")

    results = search_collection(collection, query, args.topk, args.rerank_pool, model_key, args.mode)
    print(format_search_results(results, args.snippet))


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
        return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Поиск OKF zvec</title>
  <style>
    body { font-family: system-ui, Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }
    main { max-width: 980px; margin: 0 auto; }
    form { display: flex; gap: 8px; margin: 20px 0; }
    input, select, button { font: inherit; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 6px; }
    input[type=text] { flex: 1; }
    input[type=number] { width: 76px; }
    button { background: #0f766e; color: white; border-color: #0f766e; cursor: pointer; }
    article { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; margin: 10px 0; }
    .meta { color: #64748b; font-size: 13px; margin-bottom: 6px; }
    .score { font-variant-numeric: tabular-nums; }
    pre { white-space: pre-wrap; margin: 8px 0 0; font: inherit; }
  </style>
</head>
<body>
<main>
  <h1>Поиск по базе OKF</h1>
  <form id="searchForm">
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
    <input id="topk" type="number" min="1" max="20" value="5">
    <button type="submit">Найти</button>
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
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const q = query.value.trim();
  if (!q) return;
  statusEl.textContent = 'Поиск...';
  resultsEl.innerHTML = '';
  const started = performance.now();
  const response = await fetch(`/search?q=${encodeURIComponent(q)}&topk=${encodeURIComponent(topk.value)}&model=${encodeURIComponent(model.value)}&mode=${encodeURIComponent(mode.value)}`);
  const data = await response.json();
  const elapsed = Math.round(performance.now() - started);
  statusEl.textContent = `${data.model} / ${data.mode}: результатов ${data.results.length}, ${elapsed} мс`;
  resultsEl.innerHTML = data.results.map((item) => `
    <article>
      <div class="meta"><span class="score">${Number(item.score).toFixed(4)}</span> | ${escapeHtml(item.path)}${item.heading ? ' | ' + escapeHtml(item.heading) : ''}</div>
      <strong>${escapeHtml(item.title || '')}</strong>
      <pre>${escapeHtml(item.text || '')}</pre>
    </article>
  `).join('');
});
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

        try:
            collection = _SEARCH_COLLECTIONS.get(model_key)
            if collection is None:
                raise RuntimeError(f"коллекция поиска не открыта для модели {model_key}")
            results = search_collection(collection, query, topk, rerank_pool, model_key, search_mode)
            self.send_json(
                200,
                {
                    "query": query,
                    "model": model_key,
                    "mode": search_mode,
                    "model_name": MODEL_CONFIGS[model_key]["name"],
                    "topk": topk,
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
    search_parser.set_defaults(func=command_search)

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
