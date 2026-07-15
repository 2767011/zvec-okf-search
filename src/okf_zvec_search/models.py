from __future__ import annotations

import gc
import json
import os
import time
import uuid
from typing import Any


from .config import (
    DEFAULT_MODEL,
    DIMENSION,
    MODEL_CONFIGS,
    ModelLoadError,
    _RUNTIME_SETTINGS_FILE,
    normalize_model_key,
)
from .state import (
    _METRICS,
    _MODELS,
    _MODEL_INFERENCE_LOCKS,
    _MODEL_LOCK,
    _SEARCH_LOCK,
    _SERVICE_STATE,
    _STATE_LOCK,
    log_event,
)


def normalize_preload_setting(value: str) -> tuple[str, list[str]]:
    normalized = value.strip().casefold()
    if normalized in ("", "none"):
        return "none", []
    if normalized in ("1", "true", "yes", "all"):
        return "all", list(MODEL_CONFIGS)
    models = list(
        dict.fromkeys(normalize_model_key(item.strip()) for item in normalized.split(",") if item.strip())
    )
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
    temporary = _RUNTIME_SETTINGS_FILE.with_name(f".{_RUNTIME_SETTINGS_FILE.name}.{uuid.uuid4().hex}.tmp")
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
            model_state.update(
                {
                    "name": MODEL_CONFIGS[model_key]["name"],
                    "loaded": True,
                    "load_seconds": round(duration, 3),
                }
            )
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
