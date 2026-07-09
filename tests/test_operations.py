import argparse
import base64
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import okf_zvec


class FakeCollection:
    stats = '{"doc_count":7}'


class FakeServer:
    def __init__(self, _address, _handler):
        self.okf_dir = None
        self.db_dir = None

    def serve_forever(self):
        return None


class OperationsTests(unittest.TestCase):
    def test_search_auth_supports_basic_bearer_and_open_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "search-token"
            headers = {}
            with mock.patch.object(okf_zvec, "_SEARCH_TOKEN_FILE", token_file):
                self.assertTrue(okf_zvec.is_search_authorized(headers))

                token_file.write_text("secret", encoding="utf-8")
                basic = base64.b64encode(b"okf:secret").decode("ascii")
                self.assertTrue(
                    okf_zvec.is_search_authorized({"Authorization": f"Basic {basic}"})
                )
                wrong_user = base64.b64encode(b"admin:secret").decode("ascii")
                self.assertFalse(
                    okf_zvec.is_search_authorized(
                        {"Authorization": f"Basic {wrong_user}"}
                    )
                )
                self.assertTrue(
                    okf_zvec.is_search_authorized({"Authorization": "Bearer secret"})
                )
                self.assertTrue(
                    okf_zvec.is_search_authorized(
                        {"X-OKF-Zvec-Search-Token": "secret"}
                    )
                )
                self.assertFalse(okf_zvec.is_search_authorized({"Authorization": "Bearer wrong"}))

    def test_service_start_does_not_load_models_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "db-active"
            for model_key in okf_zvec.MODEL_CONFIGS:
                (active / model_key).mkdir(parents=True)
            args = argparse.Namespace(
                db=str(root / "db"),
                okf=str(root / "okf"),
                host="127.0.0.1",
                port=8765,
            )

            with (
                mock.patch.object(okf_zvec, "read_active_db_root", return_value=active),
                mock.patch.object(okf_zvec.zvec, "open", return_value=FakeCollection()),
                mock.patch.object(okf_zvec, "ThreadingHTTPServer", FakeServer),
                mock.patch.object(okf_zvec, "get_model") as get_model,
                mock.patch.dict("os.environ", {"OKF_ZVEC_PRELOAD_MODELS": ""}),
            ):
                okf_zvec.command_serve(args)

            get_model.assert_not_called()

    def test_prometheus_output_contains_search_and_cache_metrics(self):
        metrics = okf_zvec.ServiceMetrics()
        metrics.record_search("e5", "hybrid", "success", 0.25)
        metrics.record_cache(True)
        with mock.patch.object(okf_zvec, "_METRICS", metrics):
            output = okf_zvec.prometheus_metrics()

        self.assertIn('okf_zvec_search_requests_total{model="e5",mode="hybrid"', output)
        self.assertIn('okf_zvec_cache_requests_total{result="hit"} 1', output)
        self.assertIn("okf_zvec_search_duration_seconds_sum", output)

    def test_structured_log_is_json_and_has_no_query_field(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            okf_zvec.log_event(
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
            with mock.patch.object(okf_zvec, "_RUNTIME_SETTINGS_FILE", settings_file):
                canonical, models = okf_zvec.save_preload_setting("e5")
                restored, restored_models = okf_zvec.configured_preload_setting()

            self.assertEqual(canonical, "e5")
            self.assertEqual(models, ["e5"])
            self.assertEqual(restored, "e5")
            self.assertEqual(restored_models, ["e5"])

    def test_status_actions_are_rendered_as_links(self):
        page = okf_zvec.SearchHandler.render_status(None)

        self.assertIn('data-action="apply"', page)
        self.assertIn('data-action="reload"', page)
        self.assertIn('data-action="restart"', page)
        self.assertNotIn("<button", page)


if __name__ == "__main__":
    unittest.main()
