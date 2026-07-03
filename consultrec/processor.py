import os
import signal
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from .config import LocalSettings
from .diarization import assign_roles
from .llm import run_llm_command
from .models import TranscriptSegment
from .prompts import load_clinical_prompt
from .render import render_markdown
from .storage import LocalRepository, SessionRecord
from .transcript import load_transcript_json, merge_semantic_segments, merge_short_segments, save_transcript_json, transcript_for_prompt


class TaskCanceled(RuntimeError):
    pass


def cancel_running_session(repo: LocalRepository, record: SessionRecord) -> SessionRecord:
    record = repo.cancel_session(record.case_id, record.session_id)
    stopped = _stop_record_processes(record)
    record.process_pid = 0
    repo.save_session(record)
    if stopped:
        repo.append_log(record, f"Stopped local process: {', '.join(str(pid) for pid in stopped)}.")
    else:
        repo.append_log(record, "No matching local process was found for this session.")
    return record


def run_transcription_stage(settings: LocalSettings, repo: LocalRepository, record: SessionRecord) -> SessionRecord:
    if not settings.whisper_command.strip():
        raise ValueError("Whisper command is not configured.")
    if record.status == "canceled":
        repo.append_log(record, "Transcription skipped because session is canceled.")
        return record

    record.status = "transcribing"
    repo.save_session(record)
    repo.append_log(record, "Starting WhisperX transcription.")

    session_dir = repo.find_session_dir(record.case_id, record.session_id)
    transcript_path = session_dir / "transcript.json"
    command = settings.whisper_command.format(
        audio=record.audio_path,
        output_dir=str(session_dir),
        transcript_json=str(transcript_path),
    )
    _run_cancelable_command(command, repo, record)
    record = repo.get_session(record.case_id, record.session_id)
    if record.status == "canceled":
        repo.append_log(record, "Transcription completed after cancellation; leaving session canceled.")
        return record
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Whisper command completed but did not create transcript JSON: {transcript_path}"
        )

    segments = merge_short_segments(load_transcript_json(transcript_path))
    save_transcript_json(transcript_path, segments)
    if settings.diarization_command.strip():
        record.status = "diarizing"
        repo.save_session(record)
        repo.append_log(record, "Starting speaker diarization.")
        segments = _run_diarization(settings, repo, record, segments)
    segments = merge_semantic_segments(segments)
    save_transcript_json(transcript_path, segments)

    record.transcript_path = str(transcript_path)
    record.status = "awaiting_review"
    record.process_pid = 0
    repo.save_session(record)
    repo.append_log(record, "Transcript is ready for role review.")
    return record


def generate_clinical_note(settings: LocalSettings, repo: LocalRepository, record: SessionRecord) -> SessionRecord:
    if not settings.llm_command.strip():
        raise ValueError("LLM command is not configured.")
    if not record.transcript_path:
        raise ValueError("Transcript is not ready.")
    if record.status == "canceled":
        repo.append_log(record, "Clinical note generation skipped because session is canceled.")
        return record

    record.status = "generating_note"
    repo.save_session(record)
    repo.append_log(record, "Starting clinical note generation.")

    segments = load_transcript_json(Path(record.transcript_path))
    prompt = load_clinical_prompt(settings.prompt_path, transcript_for_prompt(segments))
    note = run_llm_command(
        settings.llm_command,
        prompt,
        repo.find_session_dir(record.case_id, record.session_id),
        on_process_start=lambda pid: _record_process_pid(repo, record, pid),
        should_cancel=lambda: repo.get_session(record.case_id, record.session_id).status == "canceled",
    )
    record = repo.get_session(record.case_id, record.session_id)
    if record.status == "canceled":
        repo.append_log(record, "Clinical note generation completed after cancellation; leaving session canceled.")
        return record

    note_path = repo.write_session_json(record, "clinical_note.json", note.to_dict())
    markdown = render_markdown(record.session_id, Path(record.audio_path), segments, note)
    markdown_path = repo.write_session_text(record, "session.md", markdown)

    record.note_json_path = str(note_path)
    record.markdown_path = str(markdown_path)
    record.status = "complete"
    record.error_message = ""
    record.process_pid = 0
    repo.save_session(record)
    repo.append_log(record, "Clinical note generation completed.")
    return record


def run_diarization_stage(settings: LocalSettings, repo: LocalRepository, record: SessionRecord) -> SessionRecord:
    if not settings.diarization_command.strip():
        raise ValueError("Speaker diarization command is not configured.")
    if not record.transcript_path:
        raise ValueError("Transcript is not ready.")
    if record.status == "canceled":
        repo.append_log(record, "Speaker diarization skipped because session is canceled.")
        return record

    record.status = "diarizing"
    record.error_message = ""
    record.note_json_path = ""
    record.markdown_path = ""
    repo.save_session(record)
    repo.append_log(record, "Starting speaker diarization.")

    transcript_path = Path(record.transcript_path)
    segments = merge_short_segments(load_transcript_json(transcript_path))
    segments = _run_diarization(settings, repo, record, segments)
    segments = merge_semantic_segments(segments)
    save_transcript_json(transcript_path, segments)

    record.status = "awaiting_review"
    record.process_pid = 0
    repo.save_session(record)
    repo.append_log(record, "Speaker diarization is ready for role review.")
    return record


def update_transcript_roles(repo: LocalRepository, record: SessionRecord, segments_payload: List[Dict[str, object]]) -> SessionRecord:
    try:
        import zhconv
    except ImportError:
        zhconv = None

    segments = []
    for item in segments_payload:
        text = str(item.get("text") or "")
        if zhconv:
            text = zhconv.convert(text, 'zh-hans')
        segments.append(
            TranscriptSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker=str(item.get("speaker") or "未确认"),
                text=text,
            )
        )
    transcript_path = Path(record.transcript_path)
    save_transcript_json(transcript_path, segments)
    repo.append_log(record, "Transcript roles were updated by user review.")
    return record


def mark_error(repo: LocalRepository, record: SessionRecord, error: Exception) -> SessionRecord:
    try:
        latest = repo.get_session(record.case_id, record.session_id)
    except FileNotFoundError:
        return record
    if latest.status == "canceled":
        repo.append_log(latest, f"Task stopped after cancellation: {error}")
        return latest
    record.status = "error"
    record.error_message = str(error)
    record.process_pid = 0
    repo.save_session(record)
    repo.append_log(record, f"ERROR: {error}")
    return record


def _run_diarization(
    settings: LocalSettings,
    repo: LocalRepository,
    record: SessionRecord,
    segments: List[TranscriptSegment],
) -> List[TranscriptSegment]:
    # Custom diarization commands can still map speaker turns back onto transcript segments,
    # but role inference is intentionally left to user review.
    return assign_roles(
        segments=segments,
        mode="preserve",
        diarization_command=settings.diarization_command,
        audio_path=Path(record.audio_path),
        work_dir=repo.find_session_dir(record.case_id, record.session_id),
    )


def _run_cancelable_command(command: str, repo: LocalRepository, record: SessionRecord) -> None:
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(shlex.split(command), stdout=stdout_file, stderr=stderr_file, text=True)
        _record_process_pid(repo, record, process.pid)
        try:
            while process.poll() is None:
                latest = repo.get_session(record.case_id, record.session_id)
                if latest.status == "canceled":
                    _terminate_pid(process.pid)
                    raise TaskCanceled("Task was canceled by user.")
                time.sleep(0.8)
            if process.returncode != 0:
                stderr_file.seek(0)
                stdout_file.seek(0)
                message = (stderr_file.read() or stdout_file.read() or "").strip()
                raise RuntimeError(message or f"Command returned non-zero exit status {process.returncode}.")
        finally:
            try:
                latest = repo.get_session(record.case_id, record.session_id)
            except FileNotFoundError:
                return
            latest.process_pid = 0
            repo.save_session(latest)


def _record_process_pid(repo: LocalRepository, record: SessionRecord, pid: int) -> None:
    latest = repo.get_session(record.case_id, record.session_id)
    latest.process_pid = pid
    repo.save_session(latest)


def _stop_record_processes(record: SessionRecord) -> List[int]:
    stopped: List[int] = []
    if record.process_pid:
        if _terminate_pid(record.process_pid):
            stopped.append(record.process_pid)

    for pid, command in _matching_session_processes(record):
        if pid not in stopped and _terminate_pid(pid):
            stopped.append(pid)
    return stopped


def _matching_session_processes(record: SessionRecord) -> List[tuple]:
    try:
        result = subprocess.run(["ps", "axo", "pid=,command="], check=True, capture_output=True, text=True)
    except Exception:
        return []
    matches = []
    audio_path = str(record.audio_path)
    session_id = str(record.session_id)
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        is_session_process = audio_path in command or session_id in command
        is_known_worker = (
            "transcribe_faster_whisper.py" in command
            or "transcribe_whisperx.py" in command
            or "generate_note_ollama.py" in command
        )
        if is_session_process and is_known_worker:
            matches.append((pid, command))
    return matches


def _terminate_pid(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False

    deadline = time.time() + 4
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False
