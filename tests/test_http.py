import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from okf_zvec_search import web as okf_zvec


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.old_service_token = okf_zvec._SERVICE_TOKEN_FILE
        self.old_search_token = okf_zvec._SEARCH_TOKEN_FILE
        self.old_admin_token = okf_zvec._ADMIN_TOKEN_FILE
        self.old_collections = okf_zvec._SEARCH_COLLECTIONS
        okf_zvec._SERVICE_TOKEN_FILE = root / "service-token"
        okf_zvec._SEARCH_TOKEN_FILE = root / "search-token"
        okf_zvec._ADMIN_TOKEN_FILE = root / "admin-token"
        okf_zvec._SEARCH_TOKEN_FILE.write_text("search-secret", encoding="utf-8")
        okf_zvec._ADMIN_TOKEN_FILE.write_text("admin-secret", encoding="utf-8")
        okf_zvec._SEARCH_COLLECTIONS = {"e5": object()}
        okf_zvec._SYNC_IN_PROGRESS.clear()

        self.server = okf_zvec.ThreadingHTTPServer(("127.0.0.1", 0), okf_zvec.SearchHandler)
        self.server.okf_dir = root / "okf"
        self.server.db_dir = root / "db"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        okf_zvec._SERVICE_TOKEN_FILE = self.old_service_token
        okf_zvec._SEARCH_TOKEN_FILE = self.old_search_token
        okf_zvec._ADMIN_TOKEN_FILE = self.old_admin_token
        okf_zvec._SEARCH_COLLECTIONS = self.old_collections
        okf_zvec._SYNC_IN_PROGRESS.clear()
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None, authorized=True):
        request_headers = dict(headers or {})
        if authorized:
            request_headers.setdefault("X-OKF-Zvec-Search-Token", "search-secret")
            request_headers.setdefault("X-OKF-Zvec-Admin-Token", "admin-secret")
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, json.loads(payload) if payload else {}

    def test_search_fails_closed_without_token(self):
        status, payload = self.request("GET", "/search?q=test", authorized=False)

        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "требуется авторизация")

    def test_anonymous_search_requires_explicit_setting(self):
        okf_zvec._SEARCH_TOKEN_FILE.unlink()
        with mock.patch.dict("os.environ", {"OKF_ZVEC_ALLOW_ANONYMOUS_SEARCH": "1"}):
            status, payload = self.request("GET", "/search?q=test&topk=0", authorized=False)

        self.assertEqual(status, 400)
        self.assertIn("topk", payload["error"])

    def test_control_action_requires_separate_admin_token(self):
        status, payload = self.request(
            "POST",
            "/actions/reload-models",
            headers={"X-OKF-Zvec-Search-Token": "search-secret"},
            authorized=False,
        )

        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "требуется токен администратора")

    def test_search_rejects_out_of_range_topk(self):
        status, payload = self.request("GET", "/search?q=test&topk=0")

        self.assertEqual(status, 400)
        self.assertIn("topk", payload["error"])

    def test_search_rejects_model_without_index(self):
        status, payload = self.request("GET", "/search?q=test&model=paraphrase")

        self.assertEqual(status, 400)
        self.assertIn("индекс модели", payload["error"])

    def test_browser_action_requires_custom_header(self):
        status, payload = self.request("POST", "/actions/reload-models")

        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "недопустимый управляющий запрос")

    def test_restart_is_rejected_during_sync(self):
        okf_zvec._SYNC_IN_PROGRESS.set()
        status, payload = self.request(
            "POST",
            "/actions/restart",
            headers={"X-OKF-Zvec-Action": "1"},
        )

        self.assertEqual(status, 409)
        self.assertIn("синхронизации", payload["error"])

    def test_model_can_be_unloaded_individually(self):
        with mock.patch.object(okf_zvec, "unload_model") as unload_model:
            status, _ = self.request(
                "POST",
                "/models/e5/unload",
                headers={"X-OKF-Zvec-Action": "1"},
            )

        self.assertEqual(status, 303)
        unload_model.assert_called_once_with("e5")

    def test_unknown_model_action_is_a_client_error(self):
        status, payload = self.request(
            "POST",
            "/models/unknown/load",
            headers={"X-OKF-Zvec-Action": "1"},
        )

        self.assertEqual(status, 400)
        self.assertIn("неизвестная модель", payload["error"])

    def test_internal_search_error_is_not_exposed(self):
        with mock.patch.object(
            okf_zvec,
            "search_collection",
            side_effect=RuntimeError("secret path /opt/private"),
        ):
            status, payload = self.request("GET", "/search?q=test")

        self.assertEqual(status, 500)
        self.assertEqual(payload["error"], "внутренняя ошибка сервиса")
        self.assertNotIn("private", json.dumps(payload))

    def test_ai_search_is_added_to_observability_history(self):
        result = {
            "rank": 1,
            "relevance": 0.91,
            "score": 0.09,
            "title": "Миграция АТС",
            "path": "topics/telephony.md",
            "heading": "",
            "text": "",
            "match_terms": [],
        }
        with (
            mock.patch.object(okf_zvec, "search_collection", return_value=[result]),
            mock.patch.object(okf_zvec, "record_ai_search") as record_history,
        ):
            status, _ = self.request(
                "GET",
                "/search?q=telephony",
                headers={"X-OKF-Zvec-Origin": "ai"},
            )

        self.assertEqual(status, 200)
        entry = record_history.call_args.args[0]
        self.assertEqual(entry["query"], "telephony")
        self.assertEqual(entry["top_relevance"], 0.91)
        self.assertEqual(entry["status"], "success")

    def test_sync_rejects_oversized_body_before_reading(self):
        okf_zvec._SERVICE_TOKEN_FILE.write_text("secret", encoding="utf-8")
        with mock.patch.dict("os.environ", {"OKF_ZVEC_MAX_SYNC_BYTES": "4"}):
            status, payload = self.request(
                "POST",
                "/sync",
                body=b"12345",
                headers={"X-OKF-Zvec-Token": "secret"},
            )

        self.assertEqual(status, 413)
        self.assertIn("Content-Length", payload["error"])


if __name__ == "__main__":
    unittest.main()
