import ast
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

SUMMARY_KEYS = (
    "咨询记录",
)

SOAP_KEYS = (
    "S",
    "O",
    "A",
    "P",
)

MISSING_VALUE = "逐字稿中未明确提及"


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

    summary = _normalize_sections(summary, SUMMARY_KEYS)
    soap = _normalize_sections(soap, SOAP_KEYS)

    return ClinicalNote(
        soap=soap,
        session_summary=summary,
        raw_text=None,
    )


def clean_json_string(s: str) -> str:
    s = s.strip()
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

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
    summary_match = re.search(r'"(?:咨询记录|本次咨询内容总结|本次会谈整体摘要)"\s*:\s*"([^"]+)"', text)
    if not summary_match:
        summary_match = re.search(r'(?:咨询记录|本次咨询内容总结|本次会谈整体摘要)\s*[:：]\s*(.+?)(?=\n\n|\n[#-*]|$)', text, flags=re.DOTALL)
        
    summary_text = summary_match.group(1).strip() if summary_match else ""
    plain_summary_items = _extract_plain_summary_items(text)
    
    # Try to extract SOAP sections
    soap_sections = {
        "S": ["S (主观感觉)", "S", "主观感觉"],
        "O": ["O (客观表现)", "O", "客观表现"],
        "A": ["A (评估分析)", "A", "评估分析"],
        "P": ["P (后续计划)", "P", "后续计划"],
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
                
    if summary_text or plain_summary_items or has_soap:
        payload = {}
        if summary_text:
            payload["session_summary"] = {"咨询记录": summary_text}
        elif plain_summary_items:
            payload["session_summary"] = {"咨询记录": plain_summary_items}
        if has_soap:
            payload["soap"] = soap_data
        return payload
        
    return None


def _extract_json(text: str):
    stripped = text.strip()
    
    cleaned = clean_json_string(stripped)
    candidates = [cleaned, stripped]
    
    fenced_cleaned = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    candidates.extend(fenced_cleaned)
    
    fenced_stripped = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    candidates.extend(fenced_stripped)
    
    candidates.extend(_balanced_json_objects(cleaned))
    candidates.extend(_balanced_json_objects(stripped))
        
    for candidate in candidates:
        payload = _loads_lenient(candidate)
        if isinstance(payload, dict):
            return payload
    return None


def _loads_lenient(candidate: str):
    variants = [
        candidate,
        clean_json_string(candidate),
        _strip_trailing_commas(clean_json_string(candidate)),
    ]
    for variant in variants:
        if not variant.strip():
            continue
        try:
            return json.loads(variant)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(variant)
            except (ValueError, SyntaxError):
                continue
    return None


def _balanced_json_objects(text: str):
    objects = []
    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_plain_summary_items(text: str) -> list:
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        return []
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    items = []
    current = ""
    for line in lines:
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        match = re.match(r"^(?:[-*]|\d+[.、])\s*(.+)$", line)
        if match:
            if current:
                items.append(current.strip())
            current = match.group(1).strip()
        elif current:
            current = f"{current} {line}".strip()
    if current:
        items.append(current.strip())
    items = [_remove_advice_tail(item) for item in items]
    items = [item for item in items if item and not _looks_like_advice(item)]
    if items:
        return items
    if len(cleaned) >= 30 and not _looks_like_json_instruction(cleaned):
        return [_remove_advice_tail(cleaned)]
    return []


def _looks_like_advice(text: str) -> bool:
    advice_markers = ("建议", "可以提供", "帮助来访者", "需要找到", "应该", "最好")
    return any(marker in text for marker in advice_markers)


def _remove_advice_tail(text: str) -> str:
    for marker in ("咨询师可以", "建议", "总的来说"):
        index = text.find(marker)
        if index > 0:
            return text[:index].strip(" ，。；;")
    return text.strip()


def _looks_like_json_instruction(text: str) -> bool:
    return "session_summary" in text or "soap" in text or "{{TRANSCRIPT}}" in text


def _normalize_sections(data: dict, required_keys: tuple) -> dict:
    normalized = {}
    for key in required_keys:
        value = data.get(key)
        if value is None:
            value = _find_by_key_fragment(data, key)
        items = [item.strip() for item in _as_list(value) if str(item).strip()]
        normalized[key] = items or [MISSING_VALUE]
    for key, value in data.items():
        if key not in normalized:
            items = [item.strip() for item in _as_list(value) if str(item).strip()]
            if items:
                normalized[key] = items
    return normalized


def _find_by_key_fragment(data: dict, target: str):
    prefix = target.split(" ", 1)[0]
    for key, value in data.items():
        if str(key).strip() == target:
            return value
        if prefix in str(key) and (target in str(key) or str(key) in target):
            return value
    return None


def _as_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]
