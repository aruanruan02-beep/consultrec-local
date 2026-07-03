import tempfile
import unittest
import importlib.util
from pathlib import Path

from consultrec.diarization import assign_roles
from consultrec.llm import parse_clinical_note
from consultrec.models import TranscriptSegment
from consultrec.prompts import load_clinical_prompt
from consultrec.transcript import merge_semantic_segments, merge_short_segments, transcript_for_prompt


class PipelineUnitTests(unittest.TestCase):
    def load_whisperx_script(self):
        path = Path(__file__).resolve().parent.parent / "scripts" / "transcribe_whisperx.py"
        spec = importlib.util.spec_from_file_location("transcribe_whisperx", path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        return module

    def test_prompt_loader_requires_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.md"
            path.write_text("内容\n{{TRANSCRIPT}}", encoding="utf-8")
            prompt = load_clinical_prompt(path, "咨询师: hello")
            self.assertIn("咨询师: hello", prompt)

    def test_transcript_prompt_has_time_and_role(self):
        text = transcript_for_prompt(
            [TranscriptSegment(start=0, end=62, speaker="来访者", text="最近睡不好")]
        )
        self.assertEqual(text, "[00:00 - 01:02] 来访者: 最近睡不好")

    def test_merge_short_segments_combines_close_fragments(self):
        segments = [
            TranscriptSegment(0, 2, "第一句", "未确认"),
            TranscriptSegment(2.2, 4, "第二句", "未确认"),
            TranscriptSegment(8, 10, "第三句", "未确认"),
        ]
        merged = merge_short_segments(segments)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].text, "第一句 第二句")
        self.assertEqual(merged[0].end, 4)

    def test_merge_semantic_segments_keeps_same_speaker_together(self):
        segments = [
            TranscriptSegment(0, 12, "我最近状态不太好", "来访者"),
            TranscriptSegment(12.4, 24, "然后睡眠也受影响", "来访者"),
            TranscriptSegment(25, 32, "可以说说最近发生了什么吗", "咨询师"),
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
        self.assertEqual([item.speaker for item in assigned], ["咨询师", "来访者"])

    def test_assign_roles_preserve_keeps_detected_voice_groups(self):
        segments = [
            TranscriptSegment(0, 1, "a", "VOICE_A"),
            TranscriptSegment(1, 2, "b", "VOICE_B"),
        ]
        assigned = assign_roles(segments, mode="preserve")
        self.assertEqual([item.speaker for item in assigned], ["VOICE_A", "VOICE_B"])

    def test_whisperx_speaker_labels_are_mapped_by_first_appearance(self):
        module = self.load_whisperx_script()
        segments = module.normalize_whisperx_segments(
            [
                {"start": 0, "end": 1, "speaker": "SPEAKER_03", "text": "你好"},
                {"start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "你好"},
                {"start": 2, "end": 3, "speaker": "SPEAKER_03", "text": "继续"},
                {"start": 3, "end": 4, "text": "听不清"},
            ]
        )
        self.assertEqual([item["speaker"] for item in segments], ["说话人 1", "说话人 2", "说话人 1", "未确认"])

    def test_parse_clinical_note_accepts_fenced_json(self):
        note = parse_clinical_note(
            '```json\n{"soap":{"S":"来访者说睡不好"},"session_summary":{"本次主题":["睡眠"]}}\n```'
        )
        self.assertEqual(note.soap["S"], ["来访者说睡不好"])
        self.assertEqual(note.session_summary["本次主题"], ["睡眠"])


if __name__ == "__main__":
    unittest.main()
