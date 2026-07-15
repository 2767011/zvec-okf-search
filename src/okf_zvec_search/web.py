from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import secrets
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import zvec

from .config import (
    DEFAULT_MAX_SYNC_BYTES,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_MODE,
    MAX_RERANK_POOL,
    MAX_SNIPPET,
    MAX_TOPK,
    MODEL_CONFIGS,
    ModelLoadError,
    RequestError,
    _ADMIN_TOKEN_FILE,
    _AI_HISTORY_LIMIT,
    _SEARCH_TOKEN_FILE,
    _SERVICE_TOKEN_FILE,
    bounded_int,
    env_int,
    make_search_options,
    normalize_model_key,
    normalize_search_mode,
)
from .indexing import build_index, keep_versions, model_db_dir, read_active_db_root, sync_okf_from_archive
from .models import (
    apply_preload_setting,
    configured_index_models,
    configured_preload_setting,
    get_model,
    load_model,
    reload_model,
    reload_preloaded_models,
    save_preload_setting,
    unload_model,
)
from .search import search_collection
from .state import (
    SERVICE_STARTED_AT,
    _METRICS,
    _MODELS,
    _MODEL_LOCK,
    _QUERY_CACHE,
    _QUERY_CACHE_LOCK,
    _QUERY_CACHE_MAX,
    _RESTART_REQUESTED,
    _SEARCH_COLLECTIONS,
    _SERVICE_STATE,
    _STATE_LOCK,
    _SYNC_IN_PROGRESS,
    _SYNC_LOCK,
    ai_history_snapshot,
    log_event,
    record_ai_search,
)


def service_default_model() -> str:
    if DEFAULT_MODEL in _SEARCH_COLLECTIONS:
        return DEFAULT_MODEL
    return next(iter(_SEARCH_COLLECTIONS), DEFAULT_MODEL)


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
    state.update(
        {
            "uptime_seconds": round(time.time() - SERVICE_STARTED_AT, 1),
            "cache_entries": cache_entries,
            "cache_limit": _QUERY_CACHE_MAX,
            "loaded_models": loaded_models,
            "retained_versions": keep_versions(),
            "search_auth_enabled": bool(search_token()),
            "preload_models": preload_setting,
            "indexed_models": list(_SEARCH_COLLECTIONS),
        }
    )
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
        lines.extend(
            [
                "# HELP okf_zvec_cache_requests_total Обращения к кэшу поиска.",
                "# TYPE okf_zvec_cache_requests_total counter",
                f'okf_zvec_cache_requests_total{{result="hit"}} {_METRICS.cache_hits}',
                f'okf_zvec_cache_requests_total{{result="miss"}} {_METRICS.cache_misses}',
                "# HELP okf_zvec_search_requests_total Поисковые запросы.",
                "# TYPE okf_zvec_search_requests_total counter",
            ]
        )
        for (model, mode, status), value in sorted(_METRICS.search_requests.items()):
            lines.append(
                f'okf_zvec_search_requests_total{{model="{model}",mode="{mode}",status="{status}"}} {value}'
            )
        lines.extend(
            [
                "# HELP okf_zvec_search_duration_seconds Время выполнения поиска.",
                "# TYPE okf_zvec_search_duration_seconds summary",
            ]
        )
        for (model, mode), value in sorted(_METRICS.search_duration_sum.items()):
            labels = f'model="{model}",mode="{mode}"'
            lines.append(f"okf_zvec_search_duration_seconds_sum{{{labels}}} {value:.6f}")
            lines.append(
                f"okf_zvec_search_duration_seconds_count{{{labels}}} "
                f"{_METRICS.search_duration_count[(model, mode)]}"
            )
        lines.extend(
            [
                "# HELP okf_zvec_sync_total Синхронизации OKF.",
                "# TYPE okf_zvec_sync_total counter",
            ]
        )
        for status, value in sorted(_METRICS.sync_total.items()):
            lines.append(f'okf_zvec_sync_total{{status="{status}"}} {value}')
        lines.extend(
            [
                "# HELP okf_zvec_sync_duration_seconds Время синхронизации.",
                "# TYPE okf_zvec_sync_duration_seconds summary",
                f"okf_zvec_sync_duration_seconds_sum {_METRICS.sync_duration_sum:.6f}",
                f"okf_zvec_sync_duration_seconds_count {_METRICS.sync_duration_count}",
                "# HELP okf_zvec_model_loaded Загружена ли модель в память.",
                "# TYPE okf_zvec_model_loaded gauge",
            ]
        )
        with _MODEL_LOCK:
            loaded_models = set(_MODELS)
        for model_key in MODEL_CONFIGS:
            loaded = 1 if model_key in loaded_models else 0
            lines.append(f'okf_zvec_model_loaded{{model="{model_key}"}} {loaded}')
        lines.extend(
            [
                "# HELP okf_zvec_model_load_seconds Время загрузки модели.",
                "# TYPE okf_zvec_model_load_seconds gauge",
            ]
        )
        for model, duration in sorted(_METRICS.model_load_seconds.items()):
            lines.append(f'okf_zvec_model_load_seconds{{model="{model}"}} {duration:.6f}')
    with _STATE_LOCK:
        lines.extend(
            [
                "# HELP okf_zvec_index_documents Число фрагментов в индексе.",
                "# TYPE okf_zvec_index_documents gauge",
            ]
        )
        for model, model_state in sorted(_SERVICE_STATE["models"].items()):
            lines.append(
                f'okf_zvec_index_documents{{model="{model}"}} {int(model_state.get("doc_count", 0))}'
            )
    return "\n".join(lines) + "\n"


class SearchHandler(BaseHTTPRequestHandler):
    server_version = "OkfZvecSearch/0.7.0"
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
        record_ai_search(
            {
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
            }
        )

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
                f'<td class="history-query">{html.escape(str(entry.get("query", "")))}</td>'
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
            float(entry["top_relevance"]) for entry in successful if entry.get("top_relevance") is not None
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
      <tbody>{"".join(history_rows)}</tbody>
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
                model_actions = f'<a href="#" data-action="model-load" data-model="{model_key}">Загрузить</a>'
            model_rows.append(
                "<tr>"
                f"<td>{html.escape(model_key)}</td>"
                f"<td>{html.escape(config['name'])}</td>"
                f"<td>{'нет индекса' if not is_indexed else ('загружена' if is_loaded else 'не загружена')}</td>"
                f"<td>{int(model_state.get('doc_count', 0))}</td>"
                f'<td class="model-row-actions">{model_actions}</td>'
                "</tr>"
            )
        preload_checkboxes = "".join(
            f'<label><input type="checkbox" name="preload_models" value="{model_key}"'
            f"{' checked' if model_key in preload_models else ''}> "
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
    <dt>Активный индекс</dt><dd>{html.escape(str(state["active_db_root"]))}</dd>
    <dt>Последняя синхронизация</dt><dd>{html.escape(str(state["last_sync_at"] or "нет"))}</dd>
    <dt>Статус синхронизации</dt><dd>{html.escape(str(state["last_sync_status"]))}</dd>
    <dt>Длительность</dt><dd>{float(state["last_sync_duration_seconds"]):.2f} с</dd>
    <dt>Версий индекса</dt><dd>{int(state["retained_versions"])}</dd>
    <dt>Записей в кэше</dt><dd>{int(state["cache_entries"])} / {int(state["cache_limit"])}</dd>
    <dt>Время работы</dt><dd>{float(state["uptime_seconds"]):.1f} с</dd>
    <dt>Авторизация поиска</dt><dd>{"включена" if state["search_auth_enabled"] else "отключена"}</dd>
  </dl>
  <h2>Модели</h2>
  <table>
    <thead><tr><th>Ключ</th><th>Модель</th><th>В памяти</th><th>Фрагментов</th><th>Действие</th></tr></thead>
    <tbody>{"".join(model_rows)}</tbody>
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
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "default_model": service_default_model(),
                    "models": {key: MODEL_CONFIGS[key]["name"] for key in _SEARCH_COLLECTIONS},
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
                        {"key": key, "name": MODEL_CONFIGS[key]["name"]} for key in _SEARCH_COLLECTIONS
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
                model_key = normalize_model_key((params.get("model") or [service_default_model()])[0])
                search_mode = normalize_search_mode((params.get("mode") or [DEFAULT_SEARCH_MODE])[0])
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
                        float(params["semantic_weight"][0]) if params.get("semantic_weight") else None
                    ),
                    fts_weight=(float(params["fts_weight"][0]) if params.get("fts_weight") else None),
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
                    "results": [{**item, "text": str(item.get("text", ""))[:snippet]} for item in results],
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
                _SERVICE_STATE.update(
                    {
                        "last_sync_at": datetime.now(timezone.utc).isoformat(),
                        "last_sync_status": "success",
                        "last_sync_duration_seconds": round(duration, 3),
                        "last_sync_error": "",
                        "active_db_root": payload["active_db_root"],
                    }
                )
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
                _SERVICE_STATE.update(
                    {
                        "last_sync_at": datetime.now(timezone.utc).isoformat(),
                        "last_sync_status": "error",
                        "last_sync_duration_seconds": round(duration, 3),
                        "last_sync_error": "внутренняя ошибка сервиса",
                    }
                )
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
    _SEARCH_COLLECTIONS.clear()
    _SEARCH_COLLECTIONS.update(collections)

    preload_value, preload_models = configured_preload_setting()
    unavailable_preloads = [model for model in preload_models if model not in collections]
    if unavailable_preloads:
        raise ValueError("предзагрузка требует включённого индекса: " + ", ".join(unavailable_preloads))
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
