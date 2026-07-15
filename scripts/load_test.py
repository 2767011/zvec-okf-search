#!/usr/bin/env python3
"""Небольшой конкурентный HTTP-тест сервиса OKF Zvec."""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener


DEFAULT_QUERIES = (
    "миграция телефонии",
    "портал поставщиков",
    "обновление кассы",
    "резервное копирование",
    "доставка товара",
)
OPENER = build_opener(ProxyHandler({}))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def search_once(
    service_url: str,
    token: str,
    query: str,
    model: str,
    mode: str,
    no_cache: bool,
) -> tuple[float, int]:
    params = urlencode({
        "q": query,
        "model": model,
        "mode": mode,
        "topk": 5,
        "rerank_pool": 50,
        "no_cache": 1 if no_cache else 0,
    })
    request = Request(f"{service_url.rstrip('/')}/search?{params}")
    if token:
        credentials = base64.b64encode(f"okf:{token}".encode()).decode("ascii")
        request.add_header("Authorization", f"Basic {credentials}")
    started = time.perf_counter()
    with OPENER.open(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return (time.perf_counter() - started) * 1000, len(payload["results"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--token-file")
    parser.add_argument("--model", default="e5")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests и concurrency должны быть больше нуля")

    token = Path(args.token_file).read_text(encoding="utf-8").strip() if args.token_file else ""
    search_once(
        args.service_url,
        token,
        DEFAULT_QUERIES[0],
        args.model,
        args.mode,
        not args.use_cache,
    )

    durations: list[float] = []
    result_counts: list[int] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                search_once,
                args.service_url,
                token,
                DEFAULT_QUERIES[index % len(DEFAULT_QUERIES)],
                args.model,
                args.mode,
                not args.use_cache,
            )
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            try:
                duration, result_count = future.result()
                durations.append(duration)
                result_counts.append(result_count)
            except Exception as exc:
                errors.append(type(exc).__name__)
    wall_seconds = time.perf_counter() - started

    report = {
        "model": args.model,
        "mode": args.mode,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "cache": args.use_cache,
        "successful": len(durations),
        "errors": len(errors),
        "error_types": sorted(set(errors)),
        "throughput_rps": round(len(durations) / wall_seconds, 2) if wall_seconds else 0,
        "latency_ms": {
            "average": round(sum(durations) / len(durations), 2) if durations else 0,
            "p50": round(percentile(durations, 0.50), 2),
            "p95": round(percentile(durations, 0.95), 2),
            "maximum": round(max(durations), 2) if durations else 0,
        },
        "empty_results": sum(count == 0 for count in result_counts),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
