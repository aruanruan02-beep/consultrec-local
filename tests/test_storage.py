import tempfile
import unittest
from pathlib import Path

from consultrec.storage import LocalRepository, enrich_session_summary, safe_id, validate_audio_filename


class StorageTests(unittest.TestCase):
    def test_safe_id_rejects_empty(self):
        with self.assertRaises(ValueError):
            safe_id("   ")

    def test_validate_audio_filename(self):
        validate_audio_filename("session.m4a")
        validate_audio_filename("session.mp3")
        validate_audio_filename("session.wav")
        with self.assertRaises(ValueError):
            validate_audio_filename("notes.txt")

    def test_case_and_session_are_created_in_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "input.m4a"
            audio.write_bytes(b"fake audio")
            repo = LocalRepository(root / "data")
            case = repo.create_or_update_case("CASE 001", "note")
            record = repo.create_session(case.case_id, "2026-07-02", audio, "input.m4a")

            self.assertEqual(case.case_id, "CASE-001")
            self.assertTrue((root / "data" / "cases" / "CASE-001" / "case.json").exists())
            session_dir = repo.find_session_dir("CASE-001", record.session_id)
            self.assertTrue((session_dir / "audio.m4a").exists())
            self.assertTrue((session_dir / "session.json").exists())

    def test_list_all_sessions_adds_progress_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "input.wav"
            audio.write_bytes(b"fake audio")
            repo = LocalRepository(root / "data")
            record = repo.create_session("CASE 002", "2026-07-02", audio, "input.wav")
            sessions = repo.list_all_sessions(case_query="002")

            self.assertEqual(sessions[0]["session_id"], record.session_id)
            self.assertEqual(sessions[0]["progress_label"], "上传完成")
            self.assertEqual(sessions[0]["next_action"], "processing")

    def test_enrich_session_summary_maps_review_action(self):
        data = enrich_session_summary({"status": "awaiting_review", "audio_path": "/tmp/audio.wav"})
        self.assertEqual(data["progress_label"], "等待角色确认")
        self.assertEqual(data["next_action"], "review")

    def test_cancel_and_delete_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "input.wav"
            audio.write_bytes(b"fake audio")
            repo = LocalRepository(root / "data")
            record = repo.create_session("CASE 003", "2026-07-02", audio, "input.wav")
            session_dir = repo.find_session_dir("CASE-003", record.session_id)

            canceled = repo.cancel_session("CASE-003", record.session_id)
            self.assertEqual(canceled.status, "canceled")
            self.assertTrue((session_dir / "process.log").exists())

            repo.delete_session("CASE-003", record.session_id)
            self.assertFalse(session_dir.exists())


if __name__ == "__main__":
    unittest.main()
