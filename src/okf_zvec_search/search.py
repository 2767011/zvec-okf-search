from __future__ import annotations

import fnmatch
import json
from typing import Any

import zvec

from .config import (
    DEFAULT_MODEL,
    DEFAULT_SEARCH_MODE,
    RRF_RANK_CONSTANT,
    SearchOptions,
    normalize_model_key,
    normalize_search_mode,
)
from .models import embed
from .state import _METRICS, _QUERY_CACHE, _QUERY_CACHE_LOCK, _QUERY_CACHE_MAX, _SEARCH_LOCK
from .text import matching_terms, normalize_fts_text, normalize_raw_fts_text


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
            expressions.append(f"filter_path LIKE {zvec_string(zvec_path_pattern(options.path_pattern))}")
        else:
            expressions.append(f"filter_path = {zvec_string(options.path_pattern.casefold())}")
    if options.tags:
        tags = ", ".join(zvec_string(tag.casefold()) for tag in options.tags)
        expressions.append(f"filter_tags CONTAIN_ALL ({tags})")
    if options.date_from:
        expressions.append('filter_timestamp != ""')
        expressions.append(f"filter_timestamp >= {zvec_string(options.date_from)}")
    if options.date_to:
        expressions.append('filter_timestamp != ""')
        upper_bound = options.date_to + "T23:59:59" if len(options.date_to) == 10 else options.date_to
        expressions.append(f"filter_timestamp <= {zvec_string(upper_bound)}")
    return " AND ".join(expressions) or None


def semantic_relevance(score: float) -> float:
    return max(0.0, min(1.0, 1.0 - score))


def fts_relevance(score: float) -> float:
    positive = max(0.0, score)
    return positive / (1.0 + positive)


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
                    match_string=(normalize_raw_fts_text(query) if raw_fts else normalize_fts_text(query))
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
    fts_results = [result for result in fts_results if result_matches_filters(doc_fields(result), options)]

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
            f"{item['rank']}. relevance={item['relevance']:.4f} score={item['score']:.4f} id={item['id']}"
        )
        lines.append(f"   title: {item.get('title', '')}")
        lines.append(f"   path:  {item.get('path', '')}")
        if item.get("heading"):
            lines.append(f"   heading: {item.get('heading', '')}")
        lines.append(f"   reason: {item.get('reason', '')}")
        if text_snippet:
            lines.append(f"   text:  {text_snippet}")
    return "\n".join(lines)
