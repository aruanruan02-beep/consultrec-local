import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}


@dataclass
class CaseRecord:
    case_id: str
    created_at: str
    updated_at: str
    note: str = ""


@dataclass
class SessionRecord:
    session_id: str
    case_id: str
    session_date: str
    status: str
    audio_path: str
    transcript_path: str = ""
    note_json_path: str = ""
    markdown_path: str = ""
    error_message: str = ""
    created_at: str = ""
    updated_at: str = ""
    process_pid: int = 0


STATUS_UI = {
    "uploaded": ("上传完成", 12, True, "processing"),
    "queued_transcription": ("排队转写", 18, True, "processing"),
    "transcribing": ("转写中", 36, True, "processing"),
    "grouping_transcript": ("整理发言轮次", 58, True, "processing"),
    "diarizing": ("整理发言轮次", 58, True, "processing"),
    "diarization_failed": ("发言轮次整理失败", 68, False, "review"),
    "awaiting_review": ("等待逐字稿校对", 68, False, "review"),
    "queued_note_generation": ("排队生成记录", 74, True, "processing"),
    "generating_note": ("生成记录中", 86, True, "processing"),
    "complete": ("已完成", 100, False, "view"),
    "error": ("失败", 100, False, "retry"),
    "canceled": ("已取消", 100, False, "view"),
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    cleaned = cleaned.strip("-_")
    if not cleaned:
        raise ValueError("ID cannot be empty.")
    return cleaned


def validate_audio_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio type '{suffix}'. Allowed: {allowed}.")


class LocalRepository:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.cases_root = data_root / "cases"

    def ensure(self) -> None:
        self.cases_root.mkdir(parents=True, exist_ok=True)

    def case_dir(self, case_id: str) -> Path:
        return self.cases_root / safe_id(case_id)

    def case_file(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "case.json"

    def create_or_update_case(self, case_id: str, note: str = "") -> CaseRecord:
        self.ensure()
        case_id = safe_id(case_id)
        path = self.case_file(case_id)
        timestamp = now_iso()
        if path.exists():
            record = self.get_case(case_id)
            record.note = note if note else record.note
            record.updated_at = timestamp
        else:
            record = CaseRecord(
                case_id=case_id,
                created_at=timestamp,
                updated_at=timestamp,
                note=note,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(path, asdict(record))
        return record

    def get_case(self, case_id: str) -> CaseRecord:
        data = self._read_json(self.case_file(case_id))
        return CaseRecord(**data)

    def list_cases(self) -> List[Dict[str, Any]]:
        self.ensure()
        cases = []
        for case_path in sorted(self.cases_root.glob("*/case.json")):
            data = self._read_json(case_path)
            case_id = data["case_id"]
            sessions = self.list_sessions(case_id)
            data["session_count"] = len(sessions)
            data["latest_session_date"] = sessions[0]["session_date"] if sessions else ""
            cases.append(data)
        return sorted(cases, key=lambda item: item.get("updated_at", ""), reverse=True)

    def sessions_root(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "sessions"

    def session_dir(self, case_id: str, session_date: str, session_id: str) -> Path:
        return self.sessions_root(case_id) / f"{session_date}_{safe_id(session_id)}"

    def create_session(
        self,
        case_id: str,
        session_date: str,
        source_audio: Path,
        original_filename: str,
    ) -> SessionRecord:
        validate_audio_filename(original_filename)
        self.create_or_update_case(case_id)
        session_id = uuid.uuid4().hex[:10]
        session_dir = self.session_dir(case_id, session_date, session_id)
        session_dir.mkdir(parents=True, exist_ok=False)
        audio_path = session_dir / f"audio{Path(original_filename).suffix.lower()}"
        shutil.copyfile(source_audio, audio_path)
        timestamp = now_iso()
        record = SessionRecord(
            session_id=session_id,
            case_id=safe_id(case_id),
            session_date=session_date,
            status="uploaded",
            audio_path=str(audio_path),
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.save_session(record)
        return record

    def save_session(self, record: SessionRecord) -> None:
        record.updated_at = now_iso()
        path = self.find_session_dir(record.case_id, record.session_id) / "session.json"
        self._write_json(path, asdict(record))
        if self.case_file(record.case_id).exists():
            case = self.get_case(record.case_id)
            case.updated_at = record.updated_at
            self._write_json(self.case_file(record.case_id), asdict(case))

    def get_session(self, case_id: str, session_id: str) -> SessionRecord:
        path = self.find_session_dir(case_id, session_id) / "session.json"
        data = self._read_json(path)
        known_fields = {field.name for field in fields(SessionRecord)}
        return SessionRecord(**{key: value for key, value in data.items() if key in known_fields})

    def cancel_session(self, case_id: str, session_id: str) -> SessionRecord:
        record = self.get_session(case_id, session_id)
        record.status = "canceled"
        record.error_message = ""
        self.save_session(record)
        self.append_log(record, "Session task was canceled by user.")
        return record

    def delete_session(self, case_id: str, session_id: str) -> None:
        session_dir = self.find_session_dir(case_id, session_id)
        shutil.rmtree(session_dir)
        if self.case_file(case_id).exists():
            case = self.get_case(case_id)
            case.updated_at = now_iso()
            self._write_json(self.case_file(case_id), asdict(case))

    def list_sessions(self, case_id: str) -> List[Dict[str, Any]]:
        root = self.sessions_root(case_id)
        if not root.exists():
            return []
        sessions = []
        for session_file in root.glob("*/session.json"):
            sessions.append(self._read_json(session_file))
        return sorted(sessions, key=lambda item: item.get("created_at", ""), reverse=True)

    def list_all_sessions(
        self,
        case_query: str = "",
        status: str = "",
    ) -> List[Dict[str, Any]]:
        self.ensure()
        sessions = []
        query = case_query.strip().lower()
        for case_file in self.cases_root.glob("*/case.json"):
            case_id = self._read_json(case_file)["case_id"]
            if query and query not in case_id.lower():
                continue
            for item in self.list_sessions(case_id):
                if status and item.get("status") != status:
                    continue
                sessions.append(enrich_session_summary(item))
        return sorted(sessions, key=lambda item: item.get("created_at", ""), reverse=True)

    def find_session_dir(self, case_id: str, session_id: str) -> Path:
        root = self.sessions_root(case_id)
        matches = list(root.glob(f"*_{safe_id(session_id)}")) if root.exists() else []
        if not matches:
            raise FileNotFoundError(f"Session not found: {case_id}/{session_id}")
        return matches[0]

    def write_session_json(self, record: SessionRecord, filename: str, data: Any) -> Path:
        path = self.find_session_dir(record.case_id, record.session_id) / filename
        self._write_json(path, data)
        return path

    def write_session_text(self, record: SessionRecord, filename: str, text: str) -> Path:
        path = self.find_session_dir(record.case_id, record.session_id) / filename
        path.write_text(text, encoding="utf-8")
        return path

    def append_log(self, record: SessionRecord, message: str) -> None:
        path = self.find_session_dir(record.case_id, record.session_id) / "process.log"
        line = f"[{now_iso()}] {message}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def read_logs(self, record: SessionRecord, limit: int = 80) -> List[str]:
        path = self.find_session_dir(record.case_id, record.session_id) / "process.log"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[-limit:]

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def enrich_session_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(data)
    label, percent, is_processing, next_action = STATUS_UI.get(
        enriched.get("status", ""),
        ("未知状态", 0, False, "none"),
    )
    enriched["progress_label"] = label
    enriched["progress_percent"] = percent
    enriched["is_processing"] = is_processing
    enriched["next_action"] = next_action
    enriched["file_name"] = Path(enriched.get("audio_path", "")).name or "录音文件"
    return enriched
