from typing import Iterable, List, Sequence, Tuple

from .models import DiarizationTurn, TranscriptSegment


ACKNOWLEDGEMENTS = {
    "嗯", "嗯嗯", "嗯嗯嗯", "对", "对对", "对对对", "好", "好的", "是", "哦", "噢", "没事",
}


def normalize_diarization_turns(turns: Iterable[DiarizationTurn]) -> List[DiarizationTurn]:
    """Keep two anonymous labels stable by their first audible occurrence."""
    labels = {}
    normalized = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        if turn.end <= turn.start:
            continue
        if turn.speaker not in labels and len(labels) < 2:
            labels[turn.speaker] = f"说话人 {len(labels) + 1}"
        speaker = labels.get(turn.speaker, "未确认")
        normalized.append(DiarizationTurn(turn.start, turn.end, speaker))
    return normalized


def build_edited_segments(
    asr_segments: Iterable[TranscriptSegment],
    diarization_turns: Sequence[DiarizationTurn],
    collapse_acknowledgements: bool = True,
) -> Tuple[List[TranscriptSegment], List[TranscriptSegment]]:
    """Map ASR micro-segments to speaker turns without inventing text splits."""
    assigned: List[TranscriptSegment] = []
    turns = list(diarization_turns)
    for source in asr_segments:
        if not source.text.strip():
            continue
        overlaps = [
            (max(0.0, min(source.end, turn.end) - max(source.start, turn.start)), turn)
            for turn in turns
        ]
        overlap, best_turn = max(overlaps, default=(0.0, None), key=lambda item: item[0])
        speaker = best_turn.speaker if best_turn and overlap > 0 else "未确认"
        crosses_boundary = any(source.start < turn.start < source.end for turn in turns)
        assigned.append(
            TranscriptSegment(source.start, source.end, source.text.strip(), speaker, crosses_boundary)
        )

    grouped = _group_consecutive_speaker_segments(assigned)
    if not collapse_acknowledgements:
        return grouped, []
    return _collapse_short_acknowledgements(grouped)


def _group_consecutive_speaker_segments(segments: Sequence[TranscriptSegment]) -> List[TranscriptSegment]:
    grouped: List[TranscriptSegment] = []
    for segment in segments:
        if (
            grouped
            and grouped[-1].speaker == segment.speaker
            and segment.start - grouped[-1].end <= 1.0
        ):
            previous = grouped[-1]
            previous.end = segment.end
            previous.text = _join_text(previous.text, segment.text)
            previous.boundary_review = previous.boundary_review or segment.boundary_review
        else:
            grouped.append(
                TranscriptSegment(segment.start, segment.end, segment.text, segment.speaker, segment.boundary_review)
            )
    return grouped


def _collapse_short_acknowledgements(
    segments: Sequence[TranscriptSegment],
) -> Tuple[List[TranscriptSegment], List[TranscriptSegment]]:
    visible: List[TranscriptSegment] = []
    suppressed: List[TranscriptSegment] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        previous = visible[-1] if visible else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        if (
            previous
            and following
            and previous.speaker == following.speaker
            and current.speaker != previous.speaker
            and _is_short_acknowledgement(current)
        ):
            suppressed.append(current)
            previous.end = following.end
            previous.text = _join_text(previous.text, following.text)
            previous.boundary_review = previous.boundary_review or following.boundary_review
            index += 2
            continue
        visible.append(current)
        index += 1
    return visible, suppressed


def _is_short_acknowledgement(segment: TranscriptSegment) -> bool:
    if segment.end - segment.start > 2.0 or any(mark in segment.text for mark in "？！?!"):
        return False
    normalized = "".join(char for char in segment.text if char not in "，。、, .!！?？\n\t ")
    return normalized in ACKNOWLEDGEMENTS


def _join_text(left: str, right: str) -> str:
    if left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return f"{left} {right}"
    return f"{left}{right}"
