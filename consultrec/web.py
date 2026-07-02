import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import LocalSettings, load_settings, save_settings, settings_from_dict
from .diarization import infer_clinical_roles
from .processor import (
    cancel_running_session,
    generate_clinical_note,
    mark_error,
    run_diarization_stage,
    run_transcription_stage,
    update_transcript_roles,
)
from .storage import LocalRepository, enrich_session_summary, validate_audio_filename
from .transcript import load_transcript_json, merge_semantic_segments


app = FastAPI(title="本地咨询录音处理系统")
BASE_DIR = Path(__file__).resolve().parent.parent


def settings() -> LocalSettings:
    return load_settings()


def repo() -> LocalRepository:
    repository = LocalRepository(settings().data_root_path)
    repository.ensure()
    return repository


@app.get("/")
def index():
    built_index = BASE_DIR / "web" / "static" / "index.html"
    if built_index.exists():
        return FileResponse(built_index)
    return FileResponse(BASE_DIR / "web" / "index.html")


@app.get("/api/config")
def get_config():
    return settings().__dict__


@app.post("/api/config")
async def update_config(payload: Dict[str, str]):
    new_settings = settings_from_dict(payload)
    save_settings(new_settings)
    return new_settings.__dict__


@app.get("/api/cases")
def list_cases():
    return {"cases": repo().list_cases()}


@app.post("/api/cases")
async def create_case(payload: Dict[str, str]):
    case_id = payload.get("case_id", "")
    note = payload.get("note", "")
    if not case_id.strip():
        raise HTTPException(status_code=400, detail="Case ID is required.")
    try:
        record = repo().create_or_update_case(case_id, note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return record.__dict__


@app.get("/api/cases/{case_id}/sessions")
def list_sessions(case_id: str):
    return {"sessions": [enrich_session_summary(item) for item in repo().list_sessions(case_id)]}


@app.get("/api/sessions")
def list_all_sessions(case: str = "", status: str = ""):
    return {"sessions": repo().list_all_sessions(case_query=case, status=status)}


@app.post("/api/sessions")
async def create_session(
    background_tasks: BackgroundTasks,
    case_id: str = Form(...),
    session_date: str = Form(""),
    note: str = Form(""),
    audio: UploadFile = File(...),
):
    try:
        session_date = session_date or date.today().isoformat()
        validate_audio_filename(audio.filename or "")
        repository = repo()
        repository.create_or_update_case(case_id, note)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename or "").suffix) as temp:
            temp.write(await audio.read())
            temp_path = Path(temp.name)
        record = repository.create_session(case_id, session_date, temp_path, audio.filename or "audio")
        temp_path.unlink(missing_ok=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    background_tasks.add_task(_transcribe_background, record.case_id, record.session_id)
    return record.__dict__


@app.get("/api/sessions/{case_id}/{session_id}")
def get_session(case_id: str, session_id: str):
    repository = repo()
    current_settings = settings()
    try:
        record = repository.get_session(case_id, session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = enrich_session_summary(record.__dict__)
    payload["diarization_configured"] = bool(current_settings.diarization_command.strip())
    if record.transcript_path and Path(record.transcript_path).exists():
        segments = infer_clinical_roles(load_transcript_json(Path(record.transcript_path)))
        payload["transcript"] = [
            item.to_dict()
            for item in merge_semantic_segments(segments)
        ]
    if record.note_json_path and Path(record.note_json_path).exists():
        payload["clinical_note"] = json.loads(Path(record.note_json_path).read_text(encoding="utf-8"))
    return payload


@app.get("/api/sessions/{case_id}/{session_id}/logs")
def get_session_logs(case_id: str, session_id: str):
    repository = repo()
    try:
        record = repository.get_session(case_id, session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"logs": repository.read_logs(record)}


@app.post("/api/sessions/{case_id}/{session_id}/review")
async def review_transcript(
    background_tasks: BackgroundTasks,
    case_id: str,
    session_id: str,
    payload: Dict[str, List[Dict[str, object]]],
):
    repository = repo()
    record = repository.get_session(case_id, session_id)
    if not record.transcript_path:
        raise HTTPException(status_code=400, detail="Transcript is not ready.")
    update_transcript_roles(repository, record, payload.get("segments", []))
    background_tasks.add_task(_generate_note_background, case_id, session_id)
    record.status = "queued_note_generation"
    repository.save_session(record)
    return record.__dict__


@app.post("/api/sessions/{case_id}/{session_id}/transcript")
async def save_transcript(
    case_id: str,
    session_id: str,
    payload: Dict[str, List[Dict[str, object]]],
):
    repository = repo()
    record = repository.get_session(case_id, session_id)
    if not record.transcript_path:
        raise HTTPException(status_code=400, detail="Transcript is not ready.")
    update_transcript_roles(repository, record, payload.get("segments", []))
    record.status = "awaiting_review"
    record.note_json_path = ""
    record.markdown_path = ""
    repository.save_session(record)
    return enrich_session_summary(record.__dict__)


@app.post("/api/sessions/{case_id}/{session_id}/transcribe")
async def retry_transcription(background_tasks: BackgroundTasks, case_id: str, session_id: str):
    background_tasks.add_task(_transcribe_background, case_id, session_id)
    record = repo().get_session(case_id, session_id)
    record.status = "queued_transcription"
    record.error_message = ""
    repo().save_session(record)
    return record.__dict__


@app.post("/api/sessions/{case_id}/{session_id}/diarize")
async def rerun_diarization(background_tasks: BackgroundTasks, case_id: str, session_id: str):
    repository = repo()
    record = repository.get_session(case_id, session_id)
    if not record.transcript_path:
        raise HTTPException(status_code=400, detail="Transcript is not ready.")
    background_tasks.add_task(_diarize_background, case_id, session_id)
    record.status = "diarizing"
    record.error_message = ""
    repository.save_session(record)
    return record.__dict__


@app.post("/api/sessions/{case_id}/{session_id}/cancel")
async def cancel_session(case_id: str, session_id: str):
    repository = repo()
    try:
        record = repository.get_session(case_id, session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return cancel_running_session(repository, record).__dict__


@app.delete("/api/sessions/{case_id}/{session_id}")
async def delete_session(case_id: str, session_id: str):
    repo().delete_session(case_id, session_id)
    return {"ok": True}


@app.post("/api/sessions/{case_id}/{session_id}/generate")
async def regenerate_note(background_tasks: BackgroundTasks, case_id: str, session_id: str):
    background_tasks.add_task(_generate_note_background, case_id, session_id)
    record = repo().get_session(case_id, session_id)
    record.status = "queued_note_generation"
    repo().save_session(record)
    return record.__dict__


@app.get("/api/sessions/{case_id}/{session_id}/markdown")
def get_markdown(case_id: str, session_id: str):
    record = repo().get_session(case_id, session_id)
    if not record.markdown_path or not Path(record.markdown_path).exists():
        raise HTTPException(status_code=404, detail="Markdown has not been generated.")
    return FileResponse(record.markdown_path, media_type="text/markdown")


def _transcribe_background(case_id: str, session_id: str) -> None:
    repository = repo()
    record = repository.get_session(case_id, session_id)
    try:
        run_transcription_stage(settings(), repository, record)
    except Exception as exc:
        mark_error(repository, record, exc)


def _generate_note_background(case_id: str, session_id: str) -> None:
    repository = repo()
    record = repository.get_session(case_id, session_id)
    try:
        generate_clinical_note(settings(), repository, record)
    except Exception as exc:
        mark_error(repository, record, exc)


def _diarize_background(case_id: str, session_id: str) -> None:
    repository = repo()
    record = repository.get_session(case_id, session_id)
    try:
        run_diarization_stage(settings(), repository, record)
    except Exception as exc:
        mark_error(repository, record, exc)


app.mount("/static", StaticFiles(directory=BASE_DIR / "web" / "static"), name="static")
