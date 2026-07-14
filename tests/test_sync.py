import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import okf_zvec


class FakeCollection:
    stats = {"doc_count": 1}


def create_okf_archive(root: Path, content: str) -> Path:
    source = root / "upload" / "okf"
    source.mkdir(parents=True)
    (source / "index.md").write_text(content, encoding="utf-8")
    archive = root / "okf.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="okf")
    return archive


class SyncTests(unittest.TestCase):
    def test_build_failure_keeps_previous_okf_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            okf_dir = root / "data" / "okf"
            okf_dir.mkdir(parents=True)
            (okf_dir / "index.md").write_text("старые данные", encoding="utf-8")
            db_root = root / "data" / "db"
            old_active = root / "data" / "db-old"
            old_active.mkdir()
            active_file = root / "data" / "active-db-root"
            active_file.write_text(str(old_active), encoding="utf-8")
            archive = create_okf_archive(root, "новые данные")
            old_collections = {"old": FakeCollection()}

            def failing_build(_okf, db_dir, model_key):
                db_dir.mkdir(parents=True, exist_ok=True)
                if model_key == "paraphrase":
                    raise RuntimeError("сбой второй модели")
                return FakeCollection(), 1

            with (
                mock.patch.object(okf_zvec, "_ACTIVE_DB_FILE", active_file),
                mock.patch.object(okf_zvec, "_SEARCH_COLLECTIONS", old_collections),
                mock.patch.object(okf_zvec, "build_index", side_effect=failing_build),
            ):
                with self.assertRaisesRegex(RuntimeError, "сбой второй модели"):
                    okf_zvec.sync_okf_from_archive(archive, okf_dir, db_root)

                self.assertEqual(
                    (okf_dir / "index.md").read_text(encoding="utf-8"),
                    "старые данные",
                )
                self.assertEqual(okf_zvec._SEARCH_COLLECTIONS, old_collections)
                self.assertEqual(active_file.read_text(encoding="utf-8"), str(old_active))
                self.assertEqual(list((root / "data").glob("db-*")), [old_active])
                self.assertFalse(list((root / "data").glob(".okf.staging-*")))

    def test_success_switches_atomically_and_cleans_old_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            okf_dir = data / "okf"
            okf_dir.mkdir(parents=True)
            (okf_dir / "index.md").write_text("старые данные", encoding="utf-8")
            db_root = data / "db"
            old_versions = []
            for index in range(4):
                version = data / f"db-old-{index}"
                version.mkdir()
                os.utime(version, ns=(index + 1, index + 1))
                old_versions.append(version)
            active_file = data / "active-db-root"
            active_file.write_text(str(old_versions[-1]), encoding="utf-8")
            archive = create_okf_archive(root, "новые данные")
            old_collections = {"old": FakeCollection()}

            def successful_build(_okf, db_dir, _model_key):
                self.assertTrue(okf_zvec._SEARCH_LOCK.acquire(blocking=False))
                okf_zvec._SEARCH_LOCK.release()
                db_dir.mkdir(parents=True, exist_ok=True)
                return FakeCollection(), 1

            with (
                mock.patch.object(okf_zvec, "_ACTIVE_DB_FILE", active_file),
                mock.patch.object(okf_zvec, "_SEARCH_COLLECTIONS", old_collections),
                mock.patch.object(okf_zvec, "build_index", side_effect=successful_build),
                mock.patch.dict(os.environ, {"OKF_ZVEC_KEEP_VERSIONS": "2"}),
            ):
                result = okf_zvec.sync_okf_from_archive(archive, okf_dir, db_root)

                self.assertTrue(result["ok"])
                self.assertEqual(
                    (okf_dir / "index.md").read_text(encoding="utf-8"),
                    "новые данные",
                )
                self.assertEqual(set(okf_zvec._SEARCH_COLLECTIONS), set(okf_zvec.MODEL_CONFIGS))
                active = Path(active_file.read_text(encoding="utf-8"))
                self.assertTrue(active.is_dir())
                self.assertEqual(len(list(data.glob("db-*"))), 2)
                self.assertEqual(len(result["cleanup"]["deleted"]), 3)
                self.assertFalse(list(data.glob(".okf.backup-*")))

    def test_pointer_failure_rolls_back_okf_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            okf_dir = data / "okf"
            okf_dir.mkdir(parents=True)
            (okf_dir / "index.md").write_text("старые данные", encoding="utf-8")
            db_root = data / "db"
            old_active = data / "db-old"
            old_active.mkdir()
            archive = create_okf_archive(root, "новые данные")
            old_collections = {"old": FakeCollection()}

            def successful_build(_okf, db_dir, _model_key):
                db_dir.mkdir(parents=True, exist_ok=True)
                return FakeCollection(), 1

            with (
                mock.patch.object(okf_zvec, "_SEARCH_COLLECTIONS", old_collections),
                mock.patch.object(okf_zvec, "build_index", side_effect=successful_build),
                mock.patch.object(okf_zvec, "read_active_db_root", return_value=old_active),
                mock.patch.object(
                    okf_zvec,
                    "write_active_db_root",
                    side_effect=OSError("не удалось записать указатель"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "не удалось записать указатель"):
                    okf_zvec.sync_okf_from_archive(archive, okf_dir, db_root)

                self.assertEqual(
                    (okf_dir / "index.md").read_text(encoding="utf-8"),
                    "старые данные",
                )
                self.assertEqual(okf_zvec._SEARCH_COLLECTIONS, old_collections)
                self.assertEqual(list(data.glob("db-*")), [old_active])
                self.assertFalse(list(data.glob(".okf.backup-*")))


if __name__ == "__main__":
    unittest.main()
