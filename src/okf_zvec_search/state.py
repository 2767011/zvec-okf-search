from __future__ import annotations

from collections import OrderedDict, deque
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any


from .config import MODEL_CONFIGS, _AI_HISTORY_FILE, _AI_HISTORY_LIMIT

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


_AI_HISTORY_LOCK = threading.Lock()


_AI_HISTORY: deque[dict[str, Any]] = deque(maxlen=_AI_HISTORY_LIMIT)


_AI_HISTORY_LOADED = False


SERVICE_STARTED_AT = time.time()


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
            self.search_duration_count[duration_key] = self.search_duration_count.get(duration_key, 0) + 1

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


def clear_query_cache() -> None:
    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE.clear()


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
