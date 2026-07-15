from __future__ import annotations

import contextlib
import gc
import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import zvec

from .config import (
    DEFAULT_INDEX_BATCH_SIZE,
    DEFAULT_KEEP_VERSIONS,
    DEFAULT_MAX_ARCHIVE_MEMBERS,
    DEFAULT_MAX_EXTRACTED_BYTES,
    DEFAULT_MODEL,
    DIMENSION,
    MODEL_CONFIGS,
    RequestError,
    _ACTIVE_DB_FILE,
    env_int,
    normalize_model_key,
)
from .models import configured_index_models, embed_many
from .state import (
    _MODELS,
    _MODEL_LOCK,
    _RESTART_REQUESTED,
    _SEARCH_COLLECTIONS,
    _SEARCH_LOCK,
    _SYNC_IN_PROGRESS,
    _SYNC_LOCK,
    clear_query_cache,
)
from .text import (
    metadata_tags,
    metadata_text,
    normalize_fts_text,
    normalize_raw_fts_text,
    section_chunks,
    split_frontmatter,
    title_from_markdown,
)


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
    candidates = [path for path in db_root.parent.glob(f"{db_root.name}-*") if path.is_dir()]
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

            _SEARCH_COLLECTIONS.clear()
            _SEARCH_COLLECTIONS.update(built)
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
