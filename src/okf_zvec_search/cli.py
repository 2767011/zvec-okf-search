from __future__ import annotations

import argparse
import base64
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import zvec

from .config import (
    DEFAULT_DB_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OKF_DIR,
    DEFAULT_SEARCH_MODE,
    MODEL_CONFIGS,
    SEARCH_MODES,
    SearchOptions,
    make_search_options,
    normalize_model_key,
    normalize_search_mode,
)
from .indexing import build_index, model_db_dir, read_active_db_root
from .models import get_model
from .search import format_search_results, search_collection
from .state import clear_query_cache
from .web import command_serve


def command_index(args: argparse.Namespace) -> None:
    okf_dir = Path(args.okf).resolve()
    db_root = Path(args.db).resolve()
    model_keys = list(MODEL_CONFIGS) if args.model == "all" else [normalize_model_key(args.model)]
    for model_key in model_keys:
        db_dir = model_db_dir(db_root, model_key)
        collection, doc_count = build_index(okf_dir, db_dir, model_key)
        print(
            f"Проиндексировано фрагментов OKF Markdown: {doc_count}; каталог: {db_dir}; модель: {model_key}"
        )
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
    params = urlencode(
        {
            "q": query,
            "topk": topk,
            "rerank_pool": rerank_pool,
            "model": model_key,
            "mode": mode,
            "semantic_weight": options.semantic_weight,
            "fts_weight": options.fts_weight,
            "min_relevance": options.min_relevance,
            "no_cache": 1,
        }
    )
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
            rows.append(
                {
                    "query": query,
                    "relevant_contains": relevant,
                    "rank": rank,
                    "relevant_ranks": ranks,
                    "recall_at_k": recall_at_k,
                    "ndcg_at_k": ndcg_at_k,
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )

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
