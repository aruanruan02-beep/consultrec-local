import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import TranscriptSegment
from .transcript import save_transcript_json


FINAL_ROLES = {"Therapist", "Client", "Unknown"}
THERAPIST_CUES = (
    "咨询",
    "访谈",
    "今天",
    "我们",
    "开始",
    "了解",
    "问卷",
    "可以",
    "能不能",
    "最近",
    "感受",
    "情况",
    "问题",
    "确认",
    "接下来",
    "谢谢",
    "欢迎",
    "参加",
    "聊",
    "提供",
    "计划",
)
CLIENT_CUES = (
    "我",
    "我的",
    "男朋友",
    "女朋友",
    "家里",
    "爸爸",
    "妈妈",
    "老师",
    "同学",
    "朋友",
    "分手",
    "焦虑",
    "睡",
    "难受",
    "觉得",
    "不知道",
    "喜欢",
    "担心",
    "害怕",
)


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
    if mode == "clinical":
        return infer_clinical_roles(segments)
    if mode == "existing":
        return _map_known_speakers(segments, therapist_speaker, client_speaker)
    if mode == "alternating":
        return _assign_alternating(segments)
    raise ValueError(f"Unsupported roles mode: {mode}")


def infer_clinical_roles(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    detected = []
    for segment in segments:
        speaker = segment.speaker or "Unknown"
        if speaker in FINAL_ROLES:
            continue
        if speaker not in detected:
            detected.append(speaker)
    if not detected:
        return segments

    scores = {speaker: _role_scores_for_speaker(segments, speaker) for speaker in detected}
    if len(detected) == 1:
        speaker = detected[0]
        therapist_score, client_score = scores[speaker]
        speaker_map = {speaker: "Therapist" if therapist_score >= client_score else "Client"}
    else:
        therapist_speaker = max(detected, key=lambda item: scores[item][0] - scores[item][1])
        remaining = [item for item in detected if item != therapist_speaker]
        client_speaker = max(remaining, key=lambda item: scores[item][1] - scores[item][0]) if remaining else None
        speaker_map = {therapist_speaker: "Therapist"}
        if client_speaker:
            speaker_map[client_speaker] = "Client"
        for speaker in detected:
            speaker_map.setdefault(speaker, "Client")

    return [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=speaker_map.get(segment.speaker or "Unknown", segment.speaker or "Unknown"),
        )
        for segment in segments
    ]


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
    subprocess.run(shlex.split(command), check=True)
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


def _role_scores_for_speaker(segments: List[TranscriptSegment], speaker: str) -> tuple:
    therapist_score = 0.0
    client_score = 0.0
    for segment in segments:
        if (segment.speaker or "Unknown") != speaker:
            continue
        text = segment.text or ""
        therapist_score += _cue_score(text, THERAPIST_CUES)
        client_score += _cue_score(text, CLIENT_CUES)
        if "？" in text or "?" in text:
            therapist_score += 2.0
        if text.startswith(("请", "可以", "能不能", "我们先", "我们今天", "接下来")):
            therapist_score += 1.5
        if text.startswith(("我", "就是我", "然后我", "因为我")):
            client_score += 1.5
        if "我的" in text or "我觉得" in text or "我不知道" in text:
            client_score += 1.0
    return therapist_score, client_score


def _cue_score(text: str, cues: tuple) -> float:
    return float(sum(text.count(cue) for cue in cues))


def _map_known_speakers(
    segments: List[TranscriptSegment],
    therapist_speaker: Optional[str],
    client_speaker: Optional[str],
) -> List[TranscriptSegment]:
    speaker_map = {}
    if therapist_speaker:
        speaker_map[therapist_speaker] = "Therapist"
    if client_speaker:
        speaker_map[client_speaker] = "Client"

    known = []
    for segment in segments:
        if segment.speaker and segment.speaker not in known:
            known.append(segment.speaker)
    if not speaker_map and len(known) == 2:
        speaker_map = {known[0]: "Therapist", known[1]: "Client"}

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
    roles = ["Therapist", "Client"]
    return [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=segment.speaker or roles[index % 2],
        )
        for index, segment in enumerate(segments)
    ]
