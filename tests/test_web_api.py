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


if __name__ == "__main__":
    unittest.main()
