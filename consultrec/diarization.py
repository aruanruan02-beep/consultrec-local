import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import TranscriptSegment
from .transcript import save_transcript_json


def assign_roles(
    segments: List[TranscriptSegment],
    mode: str,
    therapist_speaker: Optional[str] = None,
    client_speaker: Optional[str] = None,
    diarization_command: Optional[str] = None,
    audio_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> List[TranscriptSegment]:
    if diarization_command:
        if not audio_path or not work_dir:
            raise ValueError("audio_path and work_dir are required for diarization command.")
        segments = _apply_external_diarization(
            segments, diarization_command, audio_path, work_dir
        )

    if mode == "preserve":
        return segments
    if mode == "existing":
        return _map_known_speakers(segments, therapist_speaker, client_speaker)
    if mode == "alternating":
        return _assign_alternating(segments)
    raise ValueError(f"Unsupported roles mode: {mode}")


def _apply_external_diarization(
    segments: List[TranscriptSegment],
    command_template: str,
    audio_path: Path,
    work_dir: Path,
) -> List[TranscriptSegment]:
    work_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = work_dir / "transcript.input.json"
    diarization_path = work_dir / "diarization.output.json"
    save_transcript_json(transcript_path, segments)
    command = command_template.format(
        audio=str(audio_path),
        transcript_json=str(transcript_path),
        diarization_json=str(diarization_path),
        output_dir=str(work_dir),
    )
    _run_command(command)
    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    speaker_turns = diarization.get("segments", diarization)
    return _merge_speaker_turns(segments, speaker_turns)


def _merge_speaker_turns(
    segments: List[TranscriptSegment],
    speaker_turns: Iterable[Dict[str, object]],
) -> List[TranscriptSegment]:
    turns = list(speaker_turns)
    merged = []
    for segment in segments:
        midpoint = (segment.start + segment.end) / 2
        speaker = segment.speaker
        for turn in turns:
            start = float(turn.get("start", 0))
            end = float(turn.get("end", 0))
            if start <= midpoint <= end:
                speaker = str(turn.get("speaker", speaker or "Unknown"))
                break
        merged.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=speaker,
            )
        )
    return merged


def _run_command(command: str) -> None:
    try:
        subprocess.run(shlex.split(command), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(message or str(exc)) from exc


def _map_known_speakers(
    segments: List[TranscriptSegment],
    therapist_speaker: Optional[str],
    client_speaker: Optional[str],
) -> List[TranscriptSegment]:
    speaker_map = {}
    if therapist_speaker:
        speaker_map[therapist_speaker] = "咨询师"
    if client_speaker:
        speaker_map[client_speaker] = "来访者"

    known = []
    for segment in segments:
        if segment.speaker and segment.speaker not in known:
            known.append(segment.speaker)
    if not speaker_map and len(known) == 2:
        speaker_map = {known[0]: "咨询师", known[1]: "来访者"}

    return [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=speaker_map.get(segment.speaker, segment.speaker or "Unknown"),
        )
        for segment in segments
    ]


def _assign_alternating(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    roles = ["咨询师", "来访者"]
    return [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=segment.speaker or roles[index % 2],
        )
        for index, segment in enumerate(segments)
    ]
