import json
from pathlib import Path
from typing import Iterable

from .models import ClinicalNote, TranscriptSegment
from .transcript import format_time


def write_outputs(
    output_dir: Path,
    session_id: str,
    audio_path: Path,
    segments: Iterable[TranscriptSegment],
    note: ClinicalNote,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_list = list(segments)
    json_payload = {
        "session_id": session_id,
        "audio_file": str(audio_path),
        "transcript": [segment.to_dict() for segment in segments_list],
        "clinical_note": note.to_dict(),
    }
    (output_dir / f"{session_id}.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{session_id}.md").write_text(
        render_markdown(session_id, audio_path, segments_list, note),
        encoding="utf-8",
    )


def render_markdown(
    session_id: str,
    audio_path: Path,
    segments: Iterable[TranscriptSegment],
    note: ClinicalNote,
) -> str:
    lines = [
        f"# Session {session_id}",
        "",
        f"- 音频文件：`{audio_path}`",
        "",
        "## Transcript",
        "",
    ]
    for segment in segments:
        lines.append(
            f"- `{format_time(segment.start)} - {format_time(segment.end)}` "
            f"**{segment.speaker or 'Unknown'}**：{segment.text}"
        )

    lines.extend(["", "## SOAP", ""])
    for key in ["S", "O", "A", "P"]:
        lines.append(f"### {key}")
        lines.extend(_bullet_list(note.soap.get(key, [])))
        lines.append("")

    lines.extend(["## Session Summary", ""])
    for key in ["本次主题", "核心问题", "情绪变化", "咨询师主要回应方式", "会谈结构", "关键转折点"]:
        lines.append(f"### {key}")
        lines.extend(_bullet_list(note.session_summary.get(key, [])))
        lines.append("")

    if note.raw_text:
        lines.extend(["## LLM Raw Output", "", "```text", note.raw_text, "```", ""])

    return "\n".join(lines)


def _bullet_list(items):
    if not items:
        return ["- 逐字稿中未明确提及"]
    return [f"- {item}" for item in items]

