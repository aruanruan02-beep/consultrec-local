import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import TranscriptSegment


def load_transcript_json(path: Path) -> List[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = _extract_segments(data)
    return [_segment_from_dict(item) for item in raw_segments]


def save_transcript_json(path: Path, segments: Iterable[TranscriptSegment]) -> None:
    payload = {"segments": [segment.to_dict() for segment in segments]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_short_segments(
    segments: Iterable[TranscriptSegment],
    target_duration: float = 12.0,
    max_duration: float = 18.0,
    max_gap: float = 1.0,
    max_chars: int = 180,
) -> List[TranscriptSegment]:
    merged: List[TranscriptSegment] = []
    for segment in segments:
        if not segment.text.strip():
            continue
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        same_speaker = _normalized_speaker(previous.speaker) == _normalized_speaker(segment.speaker)
        close_enough = segment.start - previous.end <= max_gap
        short_enough = previous.end - previous.start < target_duration
        within_limits = segment.end - previous.start <= max_duration and len(previous.text) + len(segment.text) <= max_chars
        if same_speaker and close_enough and short_enough and within_limits:
            previous.end = segment.end
            previous.text = f"{previous.text.rstrip()} {segment.text.strip()}".strip()
        else:
            merged.append(segment)
    return merged


def merge_semantic_segments(
    segments: Iterable[TranscriptSegment],
    target_duration: float = 35.0,
    max_duration: float = 55.0,
    max_gap: float = 1.2,
    max_chars: int = 520,
) -> List[TranscriptSegment]:
    merged: List[TranscriptSegment] = []
    for segment in merge_short_segments(segments, target_duration=18.0, max_duration=24.0, max_chars=260):
        if not segment.text.strip():
            continue
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        same_speaker = _normalized_speaker(previous.speaker) == _normalized_speaker(segment.speaker)
        close_enough = segment.start - previous.end <= max_gap
        within_limits = segment.end - previous.start <= max_duration and len(previous.text) + len(segment.text) <= max_chars
        should_continue = previous.end - previous.start < target_duration or _looks_like_continuation(segment.text)
        if same_speaker and close_enough and within_limits and should_continue:
            previous.end = segment.end
            previous.text = f"{previous.text.rstrip()} {segment.text.strip()}".strip()
        else:
            merged.append(segment)
    return merged


def transcript_for_prompt(segments: Iterable[TranscriptSegment]) -> str:
    lines = []
    for segment in segments:
        role = segment.speaker or "Unknown"
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


def _extract_segments(data: Any) -> List[Dict[str, Any]]:
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
        text=str(text).strip(),
        speaker=str(speaker).strip() if speaker else None,
    )


def _normalized_speaker(value: str) -> str:
    speaker = (value or "Unknown").strip()
    return speaker or "Unknown"


def _looks_like_continuation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    continuation_starts = (
        "然后",
        "所以",
        "但是",
        "不过",
        "因为",
        "就是",
        "而且",
        "同时",
        "以及",
        "那",
        "嗯",
        "对",
        "好",
        "还有",
        "另外",
        "接着",
        "后来",
    )
    return stripped.startswith(continuation_starts)
