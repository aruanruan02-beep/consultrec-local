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
        payload = _fallback_regex_extract(text)

    if payload is None:
        return ClinicalNote(
            soap={},
            session_summary={},
            raw_text=text.strip(),
        )

    soap = {}
    raw_soap = payload.get("soap", {})
    if isinstance(raw_soap, dict):
        for k, v in raw_soap.items():
            soap[k] = _as_list(v)

    summary = {}
    raw_summary = payload.get("session_summary", {})
    if isinstance(raw_summary, dict):
        for k, v in raw_summary.items():
            summary[k] = _as_list(v)

    if not soap and not summary:
        for k, v in payload.items():
            summary[k] = _as_list(v)

    return ClinicalNote(
        soap=soap,
        session_summary=summary,
        raw_text=None,
    )


def clean_json_string(s: str) -> str:
    # 1. Replace unquoted text in lists like [逐字稿中未明确提及] -> ["逐字稿中未明确提及"]
    s = re.sub(r'\[\s*([^"\'\]\s,]+)\s*\]', r'["\1"]', s)
    
    # 2. Fix key wrapped in array brackets: ["S (主观感觉)"]: -> "S (主观感觉)":
    s = re.sub(r'\[\s*("[^"]+")\s*\]\s*:', r'\1:', s)
    
    # 3. Fix "soap": [ ... ] to "soap": { ... } if it contains key-value colons
    if '"soap":' in s or '"soap"\s*:' in s:
        match = re.search(r'"soap"\s*:\s*\[', s)
        if match:
            start_idx = match.end() - 1
            depth = 1
            end_idx = -1
            for i in range(start_idx + 1, len(s)):
                if s[i] == '[':
                    depth += 1
                elif s[i] == ']':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx != -1:
                inner_content = s[start_idx+1 : end_idx]
                if ':' in inner_content:
                    s = s[:start_idx] + '{' + inner_content + '}' + s[end_idx+1:]
    return s


def _fallback_regex_extract(text: str) -> Optional[dict]:
    # Extract "本次会谈整体摘要"
    summary_match = re.search(r'"本次会谈整体摘要"\s*:\s*"([^"]+)"', text)
    if not summary_match:
        summary_match = re.search(r'本次会谈整体摘要\s*[:：]\s*(.+?)(?=\n\n|\n[#-*]|$)', text, flags=re.DOTALL)
        
    summary_text = summary_match.group(1).strip() if summary_match else ""
    
    # Try to extract SOAP sections
    soap_sections = {
        "S (主观感觉)": ["S (主观感觉)", "S", "主观感觉"],
        "O (客观表现)": ["O (客观表现)", "O", "客观表现"],
        "A (评估分析)": ["A (评估分析)", "A", "评估分析"],
        "P (后续计划)": ["P (后续计划)", "P", "后续计划"]
    }
    
    soap_data = {}
    has_soap = False
    
    for key, aliases in soap_sections.items():
        for alias in aliases:
            # 1. JSON-like array pattern
            pattern = rf'"{re.escape(alias)}"\s*:\s*\[(.*?)\]'
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                items_str = match.group(1)
                items = re.findall(r'"([^"]+)"', items_str)
                if not items:
                    # Fallback for unquoted items, split by comma
                    items = [item.strip().strip('"\'') for item in items_str.split(",") if item.strip()]
                if items:
                    soap_data[key] = items
                    has_soap = True
                    break
            
            # 2. Plain text list pattern
            pattern_text = rf'(?:{re.escape(alias)}|"{re.escape(alias)}")\s*[:：]\s*(.+?)(?=\n\n|\n[#-*]|\n\w+\s*[:：]|$)'
            match_text = re.search(pattern_text, text, flags=re.DOTALL)
            if match_text:
                content = match_text.group(1).strip()
                bullets = re.findall(r'(?:[-*•]|\d+\.)\s*(.+)', content)
                if bullets:
                    soap_data[key] = [b.strip() for b in bullets]
                else:
                    items = [item.strip().strip('"\'') for item in re.split(r'[，,]', content) if item.strip()]
                    soap_data[key] = items if items else [content.strip('"\'')]
                has_soap = True
                break
                
    if summary_text or has_soap:
        payload = {}
        if summary_text:
            payload["session_summary"] = {"本次会谈整体摘要": summary_text}
        if has_soap:
            payload["soap"] = soap_data
        return payload
        
    return None


def _extract_json(text: str):
    stripped = text.strip()
    
    # Try parsing cleaned string first
    cleaned = clean_json_string(stripped)
    candidates = [cleaned, stripped]
    
    # Try finding fenced block in cleaned
    fenced_cleaned = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    candidates.extend(fenced_cleaned)
    
    # Try finding fenced block in original
    fenced_stripped = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    candidates.extend(fenced_stripped)
    
    # Try finding braces in cleaned
    brace_cleaned = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if brace_cleaned:
        candidates.append(brace_cleaned.group(1))
        
    # Try finding braces in original
    brace_stripped = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if brace_stripped:
        candidates.append(brace_stripped.group(1))
        
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(clean_json_string(candidate))
            except json.JSONDecodeError:
                continue
    return None


def _as_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]
