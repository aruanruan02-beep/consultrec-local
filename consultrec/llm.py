import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from .models import ClinicalNote


DEFAULT_NOTE = {
    "soap": {"S": [], "O": [], "A": [], "P": []},
    "session_summary": {
        "本次主题": [],
        "核心问题": [],
        "情绪变化": [],
        "咨询师主要回应方式": [],
        "会谈结构": [],
        "关键转折点": [],
    },
}


def run_llm_command(
    command_template: str,
    prompt: str,
    work_dir: Path,
    on_process_start: Optional[Callable[[int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> ClinicalNote:
    work_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = work_dir / "clinical_note.prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    command = command_template.format(
        prompt_file=str(prompt_file),
        output_dir=str(work_dir),
        transcript_json=str(work_dir / "transcript.json"),
    )
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if on_process_start:
        on_process_start(process.pid)
    while process.poll() is None:
        if should_cancel and should_cancel():
            process.terminate()
            raise RuntimeError("Task was canceled by user.")
        time.sleep(0.8)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, shlex.split(command), stdout, stderr)
    return parse_clinical_note(stdout)


def parse_clinical_note(text: str) -> ClinicalNote:
    payload = _extract_json(text)
    if payload is None:
        note = DEFAULT_NOTE.copy()
        return ClinicalNote(
            soap=note["soap"],
            session_summary=note["session_summary"],
            raw_text=text.strip(),
        )
    return ClinicalNote(
        soap=_ensure_soap(payload.get("soap", {})),
        session_summary=_ensure_summary(payload.get("session_summary", {})),
        raw_text=None,
    )


def _extract_json(text: str):
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    candidates.extend(fenced)
    brace = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _ensure_soap(value):
    return {
        key: _as_list(value.get(key, [])) if isinstance(value, dict) else []
        for key in ["S", "O", "A", "P"]
    }


def _ensure_summary(value):
    keys = ["本次主题", "核心问题", "情绪变化", "咨询师主要回应方式", "会谈结构", "关键转折点"]
    return {
        key: _as_list(value.get(key, [])) if isinstance(value, dict) else []
        for key in keys
    }


def _as_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]
