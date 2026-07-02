import tempfile
import unittest
from pathlib import Path

from consultrec.diarization import assign_roles
from consultrec.llm import parse_clinical_note
from consultrec.models import TranscriptSegment
from consultrec.prompts import load_clinical_prompt
from consultrec.transcript import merge_semantic_segments, merge_short_segments, transcript_for_prompt


class PipelineUnitTests(unittest.TestCase):
    def test_prompt_loader_requires_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.md"
            path.write_text("内容\n{{TRANSCRIPT}}", encoding="utf-8")
            prompt = load_clinical_prompt(path, "Therapist: hello")
            self.assertIn("Therapist: hello", prompt)

    def test_transcript_prompt_has_time_and_role(self):
        text = transcript_for_prompt(
            [TranscriptSegment(start=0, end=62, speaker="Client", text="最近睡不好")]
        )
        self.assertEqual(text, "[00:00 - 01:02] Client: 最近睡不好")

    def test_merge_short_segments_combines_close_fragments(self):
        segments = [
            TranscriptSegment(0, 2, "第一句", "Unknown"),
            TranscriptSegment(2.2, 4, "第二句", "Unknown"),
            TranscriptSegment(8, 10, "第三句", "Unknown"),
        ]
        merged = merge_short_segments(segments)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].text, "第一句 第二句")
        self.assertEqual(merged[0].end, 4)

    def test_merge_semantic_segments_keeps_same_speaker_together(self):
        segments = [
            TranscriptSegment(0, 12, "我最近状态不太好", "Client"),
            TranscriptSegment(12.4, 24, "然后睡眠也受影响", "Client"),
            TranscriptSegment(25, 32, "可以说说最近发生了什么吗", "Therapist"),
        ]
        merged = merge_semantic_segments(segments)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].text, "我最近状态不太好 然后睡眠也受影响")

    def test_assign_roles_maps_two_known_speakers(self):
        segments = [
            TranscriptSegment(0, 1, "a", "SPEAKER_01"),
            TranscriptSegment(1, 2, "b", "SPEAKER_02"),
        ]
        assigned = assign_roles(segments, mode="existing")
        self.assertEqual([item.speaker for item in assigned], ["Therapist", "Client"])

    def test_assign_roles_preserve_keeps_detected_voice_groups(self):
        segments = [
            TranscriptSegment(0, 1, "a", "VOICE_A"),
            TranscriptSegment(1, 2, "b", "VOICE_B"),
        ]
        assigned = assign_roles(segments, mode="preserve")
        self.assertEqual([item.speaker for item in assigned], ["VOICE_A", "VOICE_B"])

    def test_assign_roles_clinical_maps_voice_groups_to_roles(self):
        segments = [
            TranscriptSegment(0, 3, "今天我们先了解一下最近的情况", "VOICE_A"),
            TranscriptSegment(4, 8, "我最近一直睡不好", "VOICE_B"),
        ]
        assigned = assign_roles(segments, mode="clinical")
        self.assertEqual([item.speaker for item in assigned], ["Therapist", "Client"])

    def test_parse_clinical_note_accepts_fenced_json(self):
        note = parse_clinical_note(
            '```json\n{"soap":{"S":"来访者说睡不好"},"session_summary":{"本次主题":["睡眠"]}}\n```'
        )
        self.assertEqual(note.soap["S"], ["来访者说睡不好"])
        self.assertEqual(note.session_summary["本次主题"], ["睡眠"])


if __name__ == "__main__":
    unittest.main()
