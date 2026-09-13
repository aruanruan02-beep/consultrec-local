import importlib.util
import tempfile
import unittest
from pathlib import Path


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed")
class WebApiTests(unittest.TestCase):
    def test_create_case_and_list_cases(self):
        from fastapi.testclient import TestClient

        from consultrec.config import LocalSettings
        from consultrec import web

        with tempfile.TemporaryDirectory() as tmp:
            web.settings = lambda: LocalSettings(data_root=str(Path(tmp) / "data"))
            client = TestClient(web.app)

            created = client.post("/api/cases", json={"case_id": "CASE 100", "note": "demo"})
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["case_id"], "CASE-100")

            listed = client.get("/api/cases")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["cases"][0]["case_id"], "CASE-100")

            sessions = client.get("/api/sessions")
            self.assertEqual(sessions.status_code, 200)
            self.assertEqual(sessions.json()["sessions"], [])

    def test_session_detail_preserves_detected_speaker_labels(self):
        from fastapi.testclient import TestClient

        from consultrec.config import LocalSettings
        from consultrec.models import TranscriptSegment
        from consultrec.storage import LocalRepository
        from consultrec.transcript import save_transcript_document
        from consultrec import web

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            web.settings = lambda: LocalSettings(data_root=str(data_root))
            repository = LocalRepository(data_root)
            repository.ensure()
            source_audio = Path(tmp) / "audio.m4a"
            source_audio.write_bytes(b"fake audio")
            record = repository.create_session("CASE 200", "2026-07-03", source_audio, "audio.m4a")
            transcript_path = repository.find_session_dir(record.case_id, record.session_id) / "transcript.json"
            save_transcript_document(
                transcript_path,
                [
                    TranscriptSegment(0, 1, "你好", "说话人 1"),
                    TranscriptSegment(1, 2, "你好", "说话人 2"),
                ],
                [
                    TranscriptSegment(0, 1, "已校对的文字", "咨询师"),
                    TranscriptSegment(1, 2, "你好", "未确认"),
                ],
            )
            record.transcript_path = str(transcript_path)
            repository.save_session(record)

            client = TestClient(web.app)
            detail = client.get(f"/api/sessions/{record.case_id}/{record.session_id}")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(
                [item["speaker"] for item in detail.json()["asr_segments"]],
                ["说话人 1", "说话人 2"],
            )
            self.assertEqual(detail.json()["edited_segments"][0]["text"], "已校对的文字")

            saved = client.post(
                f"/api/sessions/{record.case_id}/{record.session_id}/transcript",
                json={
                    "segments": [
                        {"start": 0, "end": 0.5, "speaker": "咨询师", "text": "拆分后的前半段"},
                        {"start": 0.5, "end": 1, "speaker": "咨询师", "text": "拆分后的后半段"},
                        {"start": 1, "end": 2, "speaker": "未确认", "text": "你好"},
                    ]
                },
            )
            self.assertEqual(saved.status_code, 200)
            updated = client.get(f"/api/sessions/{record.case_id}/{record.session_id}").json()
            self.assertEqual(len(updated["asr_segments"]), 2)
            self.assertEqual(len(updated["edited_segments"]), 3)

    def test_save_clinical_note_endpoint(self):
        from fastapi.testclient import TestClient
        from consultrec.config import LocalSettings
        from consultrec.storage import LocalRepository
        from consultrec.transcript import save_transcript_json
        from consultrec.models import TranscriptSegment
        from consultrec import web

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            web.settings = lambda: LocalSettings(data_root=str(data_root))
            repository = LocalRepository(data_root)
            repository.ensure()
            source_audio = Path(tmp) / "audio.m4a"
            source_audio.write_bytes(b"fake audio")
            record = repository.create_session("CASE 300", "2026-07-03", source_audio, "audio.m4a")
            transcript_path = repository.find_session_dir(record.case_id, record.session_id) / "transcript.json"
            save_transcript_json(
                transcript_path,
                [TranscriptSegment(0, 1, "你好", "说话人 1")],
            )
            record.transcript_path = str(transcript_path)
            repository.save_session(record)

            client = TestClient(web.app)
            payload = {
                "session_summary": {"本次会谈整体摘要": ["摘要文本"]},
                "soap": {
                    "S (主观感觉)": ["感觉不错"],
                    "O (客观表现)": ["逐字稿中未明确提及"],
                    "A (评估分析)": ["逐字稿中未明确提及"],
                    "P (后续计划)": ["逐字稿中未明确提及"]
                }
            }
            res = client.post(f"/api/sessions/{record.case_id}/{record.session_id}/note", json=payload)
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["clinical_note"]["session_summary"]["本次会谈整体摘要"], ["摘要文本"])


if __name__ == "__main__":
    unittest.main()
