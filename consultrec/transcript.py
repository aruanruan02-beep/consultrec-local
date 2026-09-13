import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import DiarizationTurn, TranscriptSegment


_ASCII_TO_FULLWIDTH_PUNCTUATION = str.maketrans({
    ",": "，",
    ".": "。",
    ":": "：",
    ";": "；",
    "?": "？",
    "!": "！",
    "(": "（",
    ")": "）",
    "[": "【",
    "]": "】",
})


def normalize_chinese_punctuation(text: str) -> str:
    """Use Chinese full-width punctuation in transcript text."""
    normalized = str(text or "")
    normalized = re.sub(r"\.{3,}", "……", normalized)
    return normalized.translate(_ASCII_TO_FULLWIDTH_PUNCTUATION)


def load_transcript_json(path: Path) -> List[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = _extract_segments(data, preferred_key="edited_segments")
    return [_segment_from_dict(item) for item in raw_segments]


def save_transcript_json(path: Path, segments: Iterable[TranscriptSegment]) -> None:
    payload = {"segments": [_segment_to_dict(segment) for segment in segments]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_transcript_document(path: Path) -> Dict[str, List[TranscriptSegment]]:
    """Load current dual-layer transcripts and transparently read legacy files."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("asr_segments"), list):
        asr_segments = [_segment_from_dict(item) for item in data["asr_segments"]]
        edited_raw = data.get("edited_segments", data.get("segments", data["asr_segments"]))
        if not isinstance(edited_raw, list):
            raise ValueError("Transcript edited_segments must be a list.")
        return {
            "asr_segments": asr_segments,
            "edited_segments": [_segment_from_dict(item) for item in edited_raw],
            "diarization_turns": [_turn_from_dict(item) for item in data.get("diarization_turns", [])],
            "suppressed_turns": [_segment_from_dict(item) for item in data.get("suppressed_turns", [])],
        }

    legacy = [_segment_from_dict(item) for item in _extract_segments(data)]
    return {"asr_segments": copy.deepcopy(legacy), "edited_segments": legacy, "diarization_turns": [], "suppressed_turns": []}


def save_transcript_document(
    path: Path,
    asr_segments: Iterable[TranscriptSegment],
    edited_segments: Iterable[TranscriptSegment],
    diarization_turns: Iterable[DiarizationTurn] = (),
    suppressed_turns: Iterable[TranscriptSegment] = (),
) -> None:
    payload = {
        "asr_segments": [_segment_to_dict(segment) for segment in asr_segments],
        "edited_segments": [_segment_to_dict(segment) for segment in edited_segments],
        "diarization_turns": [turn.to_dict() for turn in diarization_turns],
        "suppressed_turns": [_segment_to_dict(segment) for segment in suppressed_turns],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def transcript_for_prompt(segments: Iterable[TranscriptSegment]) -> str:
    lines = []
    for segment in segments:
        role = segment.speaker or "未确认"
        lines.append(
            f"[{format_time(segment.start)} - {format_time(segment.end)}] "
            f"{role}: {segment.text}"
        )
    return "\n".join(lines)


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _extract_segments(data: Any, preferred_key: str = "segments") -> List[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get(preferred_key), list):
        return data[preferred_key]
    if isinstance(data, dict) and preferred_key != "segments" and isinstance(data.get("segments"), list):
        return data["segments"]
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError("Transcript JSON must contain a top-level 'segments' list.")


def _segment_from_dict(item: Dict[str, Any]) -> TranscriptSegment:
    start = item.get("start", item.get("start_time", item.get("from")))
    end = item.get("end", item.get("end_time", item.get("to")))
    text = item.get("text", item.get("sentence", ""))
    speaker = item.get("speaker", item.get("role"))
    if start is None or end is None:
        raise ValueError(f"Transcript segment is missing start/end: {item}")
    return TranscriptSegment(
        start=float(start),
        end=float(end),
        text=normalize_chinese_punctuation(str(text).strip()),
        speaker=str(speaker).strip() if speaker else None,
        boundary_review=bool(item.get("boundary_review", False)),
    )


def _segment_to_dict(segment: TranscriptSegment) -> Dict[str, Any]:
    payload = segment.to_dict()
    payload["text"] = normalize_chinese_punctuation(segment.text).strip()
    return payload


def _turn_from_dict(item: Dict[str, Any]) -> DiarizationTurn:
    start = item.get("start")
    end = item.get("end")
    if start is None or end is None:
        raise ValueError(f"Diarization turn is missing start/end: {item}")
    return DiarizationTurn(start=float(start), end=float(end), speaker=str(item.get("speaker") or "未确认"))
