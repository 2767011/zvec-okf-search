import argparse
import base64
import io
import json
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from okf_zvec_search import models, state, web


class FakeCollection:
    stats = '{"doc_count":7}'


class FakeServer:
    def __init__(self, _address, _handler):
        self.okf_dir = None
        self.db_dir = None

    def serve_forever(self):
        return None

    def server_close(self):
        return None


class OperationsTests(unittest.TestCase):
    def test_index_models_are_configurable(self):
        with mock.patch.dict("os.environ", {"OKF_ZVEC_INDEX_MODELS": "e5"}):
            self.assertEqual(models.configured_index_models(), ["e5"])

    def test_home_only_offers_indexed_models(self):
        with mock.patch.dict(web._SEARCH_COLLECTIONS, {"e5": object()}, clear=True):
            page = web.SearchHandler.render_home(None)

        self.assertIn('value="e5"', page)
        self.assertNotIn('value="paraphrase"', page)

    def test_search_auth_supports_basic_bearer_and_explicit_anonymous_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "search-token"
            headers = {}
            with mock.patch.object(web, "_SEARCH_TOKEN_FILE", token_file):
                self.assertFalse(web.is_search_authorized(headers))
                with mock.patch.dict("os.environ", {"OKF_ZVEC_ALLOW_ANONYMOUS_SEARCH": "true"}):
                    self.assertTrue(web.is_search_authorized(headers))

                token_file.write_text("secret", encoding="utf-8")
                basic = base64.b64encode(b"okf:secret").decode("ascii")
                self.assertTrue(web.is_search_authorized({"Authorization": f"Basic {basic}"}))
                wrong_user = base64.b64encode(b"admin:secret").decode("ascii")
                self.assertFalse(web.is_search_authorized({"Authorization": f"Basic {wrong_user}"}))
                self.assertTrue(web.is_search_authorized({"Authorization": "Bearer secret"}))
                self.assertTrue(web.is_search_authorized({"X-OKF-Zvec-Search-Token": "secret"}))
                self.assertFalse(web.is_search_authorized({"Authorization": "Bearer wrong"}))

    def test_service_start_does_not_load_models_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "db-active"
            for model_key in web.MODEL_CONFIGS:
                (active / model_key).mkdir(parents=True)
            args = argparse.Namespace(
                db=str(root / "db"),
                okf=str(root / "okf"),
                host="127.0.0.1",
                port=8765,
            )

            with (
                mock.patch.object(web, "read_active_db_root", return_value=active),
                mock.patch.object(web.zvec, "open", return_value=FakeCollection()),
                mock.patch.object(web, "ThreadingHTTPServer", FakeServer),
                mock.patch.object(web, "get_model") as get_model,
                mock.patch.dict("os.environ", {"OKF_ZVEC_PRELOAD_MODELS": ""}),
            ):
                web.command_serve(args)

            get_model.assert_not_called()

    def test_prometheus_output_contains_search_and_cache_metrics(self):
        metrics = state.ServiceMetrics()
        metrics.record_search("e5", "hybrid", "success", 0.25)
        metrics.record_cache(True)
        with mock.patch.object(web, "_METRICS", metrics):
            output = web.prometheus_metrics()

        self.assertIn('okf_zvec_search_requests_total{model="e5",mode="hybrid"', output)
        self.assertIn('okf_zvec_cache_requests_total{result="hit"} 1', output)
        self.assertIn("okf_zvec_search_duration_seconds_sum", output)

    def test_structured_log_is_json_and_has_no_query_field(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            state.log_event(
                "search_completed",
                model="e5",
                mode="hybrid",
                result_count=3,
            )
        payload = json.loads(stream.getvalue())

        self.assertEqual(payload["event"], "search_completed")
        self.assertEqual(payload["result_count"], 3)
        self.assertNotIn("query", payload)

    def test_web_preload_setting_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "runtime-settings.json"
            with mock.patch.object(models, "_RUNTIME_SETTINGS_FILE", settings_file):
                canonical, selected_models = models.save_preload_setting("e5")
                restored, restored_models = models.configured_preload_setting()

            self.assertEqual(canonical, "e5")
            self.assertEqual(selected_models, ["e5"])
            self.assertEqual(restored, "e5")
            self.assertEqual(restored_models, ["e5"])

    def test_status_actions_are_rendered_as_links(self):
        with mock.patch.dict(web._MODELS, {"e5": object()}, clear=True):
            page = web.SearchHandler.render_status(None)

        self.assertIn('data-action="save-preload"', page)
        self.assertIn('data-action="model-load"', page)
        self.assertIn('data-action="model-unload"', page)
        self.assertIn('data-action="restart"', page)
        self.assertIn('href="/ai-history"', page)
        self.assertNotIn("Последние запросы ИИ", page)
        self.assertNotIn("<button", page)

    def test_ai_history_has_a_separate_page(self):
        history = [
            {
                "timestamp": "2026-07-15T10:00:00+05:00",
                "query": "переезд телефонии",
                "model": "e5",
                "mode": "hybrid",
                "duration_ms": 42,
                "result_count": 3,
                "top_title": "Миграция АТС",
                "top_path": "topics/telephony.md",
                "top_relevance": 0.91,
                "status": "success",
            }
        ]
        with mock.patch.object(web, "ai_history_snapshot", return_value=history):
            page = web.SearchHandler.render_ai_history(None)

        self.assertIn("Последние запросы ИИ", page)
        self.assertIn("переезд телефонии", page)
        self.assertIn("91%", page)
        self.assertIn("В выборке: 1 из 20", page)

    def test_ai_history_is_persistent_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            old_history = state._AI_HISTORY
            old_loaded = state._AI_HISTORY_LOADED
            try:
                state._AI_HISTORY = deque(maxlen=state._AI_HISTORY_LIMIT)
                state._AI_HISTORY_LOADED = False
                with mock.patch.object(state, "_AI_HISTORY_FILE", history_file):
                    for number in range(22):
                        state.record_ai_search({"query": f"query-{number}"})
                    history = state.ai_history_snapshot()

                self.assertEqual(len(history), 20)
                self.assertEqual(history[0]["query"], "query-21")
                self.assertEqual(history[-1]["query"], "query-2")
                self.assertEqual(len(json.loads(history_file.read_text(encoding="utf-8"))), 20)
            finally:
                state._AI_HISTORY = old_history
                state._AI_HISTORY_LOADED = old_loaded


if __name__ == "__main__":
    unittest.main()
