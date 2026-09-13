import os
import selectors
import signal
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from .config import LocalSettings
from .llm import run_llm_command
from .models import DiarizationTurn, TranscriptSegment
from .prompts import load_clinical_prompt
from .render import render_markdown
from .storage import LocalRepository, SessionRecord
from .transcript import (
    load_transcript_document,
    load_transcript_json,
    save_transcript_document,
    transcript_for_prompt,
)
from .turns import build_edited_segments, normalize_diarization_turns


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
    repo.append_log(record, "Starting local ASR transcription.")

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

    asr_segments = load_transcript_json(transcript_path)
    for segment in asr_segments:
        segment.speaker = "未确认"
    save_transcript_document(transcript_path, asr_segments, asr_segments)
    record.transcript_path = str(transcript_path)
    record.status = "diarizing"
    repo.save_session(record)
    repo.append_log(record, "ASR transcript is ready. Starting local speaker-turn diarization.")
    return run_diarization_stage(settings, repo, record, automatic=True)


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

    segments = load_transcript_document(Path(record.transcript_path))["edited_segments"]
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


def run_diarization_stage(
    settings: LocalSettings, repo: LocalRepository, record: SessionRecord, automatic: bool = False
) -> SessionRecord:
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
    repo.append_log(record, "Starting local speaker-turn diarization.")

    transcript_path = Path(record.transcript_path)
    document = load_transcript_document(transcript_path)
    turns = _run_diarization(settings, repo, record)
    turns = normalize_diarization_turns(turns)
    segments, suppressed_turns = build_edited_segments(document["asr_segments"], turns)
    save_transcript_document(transcript_path, document["asr_segments"], segments, turns, suppressed_turns)

    record.status = "awaiting_review"
    record.process_pid = 0
    repo.save_session(record)
    repo.append_log(record, f"Speaker-turn transcript is ready for review ({len(segments)} segments).")
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
    _validate_edited_segments(segments)
    transcript_path = Path(record.transcript_path)
    document = load_transcript_document(transcript_path)
    save_transcript_document(
        transcript_path, document["asr_segments"], segments, document["diarization_turns"], document["suppressed_turns"]
    )
    repo.append_log(record, "Edited transcript was saved by user review.")
    return record


def restore_short_responses(repo: LocalRepository, record: SessionRecord) -> SessionRecord:
    if not record.transcript_path:
        raise ValueError("Transcript is not ready.")
    transcript_path = Path(record.transcript_path)
    document = load_transcript_document(transcript_path)
    if not document["diarization_turns"]:
        raise ValueError("Speaker-turn diarization is not available.")
    segments, _ = build_edited_segments(
        document["asr_segments"], document["diarization_turns"], collapse_acknowledgements=False
    )
    save_transcript_document(transcript_path, document["asr_segments"], segments, document["diarization_turns"], [])
    repo.append_log(record, "Short acknowledgement turns were restored for review.")
    return record


def _validate_edited_segments(segments: List[TranscriptSegment]) -> None:
    previous_end = -1.0
    for segment in segments:
        if segment.end <= segment.start:
            raise ValueError("Transcript segment end time must be after its start time.")
        if segment.start < previous_end - 0.001:
            raise ValueError("Transcript segment times must not overlap.")
        previous_end = segment.end


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


def _run_diarization(settings: LocalSettings, repo: LocalRepository, record: SessionRecord) -> List[DiarizationTurn]:
    session_dir = repo.find_session_dir(record.case_id, record.session_id)
    output_path = session_dir / "diarization.json"
    command = settings.diarization_command.format(
        audio=record.audio_path,
        diarization_json=str(output_path),
        output_dir=str(session_dir),
    )
    _run_cancelable_command(command, repo, record)
    if not output_path.exists():
        raise FileNotFoundError("Diarization command completed without creating diarization JSON.")
    payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    turns = payload.get("turns", payload.get("segments", []))
    return [DiarizationTurn(float(item["start"]), float(item["end"]), str(item["speaker"])) for item in turns]


def _run_cancelable_command(command: str, repo: LocalRepository, record: SessionRecord) -> None:
    output_lines: List[str] = []
    selector = selectors.DefaultSelector()
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        if process.stdout:
            selector.register(process.stdout, selectors.EVENT_READ)
        _record_process_pid(repo, record, process.pid)
        while process.poll() is None:
            latest = repo.get_session(record.case_id, record.session_id)
            if latest.status == "canceled":
                _terminate_pid(process.pid)
                raise TaskCanceled("Task was canceled by user.")
            for key, _ in selector.select(timeout=0.8):
                line = key.fileobj.readline()
                if line:
                    clean_line = line.rstrip()
                    output_lines.append(clean_line)
                    repo.append_log(record, clean_line)

        if process.stdout:
            for line in process.stdout:
                clean_line = line.rstrip()
                if clean_line:
                    output_lines.append(clean_line)
                    repo.append_log(record, clean_line)

        if process.returncode != 0:
            message = "\n".join(line for line in output_lines if line).strip()
            raise RuntimeError(message or f"Command returned non-zero exit status {process.returncode}.")
    finally:
        selector.close()
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
            "transcribe_mlx_whisper.py" in command
            or "generate_note_ollama.py" in command
            or "diarize_pyannote.py" in command
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
