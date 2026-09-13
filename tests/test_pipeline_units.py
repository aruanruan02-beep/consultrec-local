import tempfile
import unittest
import importlib.util
from pathlib import Path

from consultrec.diarization import assign_roles
from consultrec.llm import parse_clinical_note
from consultrec.models import DiarizationTurn, TranscriptSegment
from consultrec.prompts import load_clinical_prompt
from consultrec.transcript import normalize_chinese_punctuation, transcript_for_prompt
from consultrec.turns import build_edited_segments, normalize_diarization_turns


class PipelineUnitTests(unittest.TestCase):
    def test_transcript_punctuation_uses_chinese_fullwidth_symbols(self):
        self.assertEqual(
            normalize_chinese_punctuation("你好,今天怎么样? 好... (测试)"),
            "你好，今天怎么样？ 好…… （测试）",
        )

    def load_note_script(self):
        path = Path(__file__).resolve().parent.parent / "scripts" / "generate_note_ollama.py"
        spec = importlib.util.spec_from_file_location("generate_note_ollama", path)
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

    def test_note_script_splits_transcript_on_line_boundaries(self):
        module = self.load_note_script()
        chunks = module.split_transcript("a" * 10 + "\n" + "b" * 10 + "\n" + "c" * 10, 15)
        self.assertEqual(chunks, ["a" * 10, "b" * 10, "c" * 10])

    def test_transcript_prompt_has_time_and_role(self):
        text = transcript_for_prompt(
            [TranscriptSegment(start=0, end=62, speaker="来访者", text="最近睡不好")]
        )
        self.assertEqual(text, "[00:00 - 01:02] 来访者: 最近睡不好")

    def test_turn_mapping_uses_maximum_overlap_and_marks_crossing_boundaries(self):
        turns = normalize_diarization_turns([
            DiarizationTurn(0, 2, "SPEAKER_01"),
            DiarizationTurn(2, 5, "SPEAKER_00"),
        ])
        edited, suppressed = build_edited_segments(
            [TranscriptSegment(0, 1, "第一句"), TranscriptSegment(1.1, 3, "第二句")], turns
        )
        self.assertEqual(len(suppressed), 0)
        self.assertEqual([item.speaker for item in edited], ["说话人 1", "说话人 2"])
        self.assertTrue(edited[1].boundary_review)

    def test_turn_mapping_collapses_only_brief_acknowledgements_between_same_speaker(self):
        turns = normalize_diarization_turns([
            DiarizationTurn(0, 2, "A"),
            DiarizationTurn(2, 3, "B"),
            DiarizationTurn(3, 5, "A"),
        ])
        edited, suppressed = build_edited_segments(
            [
                TranscriptSegment(0, 2, "我最近睡不好。"),
                TranscriptSegment(2, 3, "嗯。"),
                TranscriptSegment(3, 5, "白天也很累。"),
            ],
            turns,
        )
        self.assertEqual(len(edited), 1)
        self.assertEqual(edited[0].text, "我最近睡不好。白天也很累。")
        self.assertEqual([item.text for item in suppressed], ["嗯。"])

    def test_turn_mapping_keeps_short_questions_visible(self):
        turns = normalize_diarization_turns([
            DiarizationTurn(0, 2, "A"),
            DiarizationTurn(2, 3, "B"),
            DiarizationTurn(3, 5, "A"),
        ])
        edited, suppressed = build_edited_segments(
            [TranscriptSegment(0, 2, "我最近睡不好。"), TranscriptSegment(2, 3, "为什么？"), TranscriptSegment(3, 5, "工作太忙。")],
            turns,
        )
        self.assertEqual(len(edited), 3)
        self.assertEqual(suppressed, [])

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

    def test_parse_clinical_note_accepts_fenced_json(self):
        note = parse_clinical_note(
            '```json\n{"soap":{"S":"来访者说睡不好"},"session_summary":{"本次主题":["睡眠"]}}\n```'
        )
        self.assertEqual(note.soap["S"], ["来访者说睡不好"])
        self.assertEqual(note.session_summary["本次主题"], ["睡眠"])

    def test_parse_clinical_note_accepts_json_with_surrounding_text(self):
        note = parse_clinical_note(
            '下面是记录：\n{"session_summary":{"本次会谈整体摘要":["来访者谈到最近睡眠受影响。"]},'
            '"soap":{"S (主观感觉)":["睡不好"],"P (后续计划)":["下次继续讨论睡眠"]}}\n已完成。'
        )
        self.assertIsNone(note.raw_text)
        self.assertEqual(note.session_summary["本次会谈整体摘要"], ["来访者谈到最近睡眠受影响。"])
        self.assertEqual(note.soap["S (主观感觉)"], ["睡不好"])
        self.assertEqual(note.soap["O"], ["逐字稿中未明确提及"])

    def test_parse_clinical_note_falls_back_to_plain_summary_list(self):
        note = parse_clinical_note(
            "从来访者的描述来看，本次主要谈到：\n\n"
            "1. **导师关系压力**：来访者谈到导师频繁分配任务，自己会担心失去信任。\n"
            "2. **具体事件**：来访者讲述博士招生面试当天没有带电脑、电源处理和记录任务带来的自责。\n"
            "3. **工作经历影响**：来访者回顾互联网工作中长期处于被评价和高压状态。\n\n"
            "咨询师可以提供一些情绪调节建议。"
        )
        self.assertIsNone(note.raw_text)
        self.assertIn("导师关系压力", note.session_summary["咨询记录"][0])
        self.assertEqual(len(note.session_summary["咨询记录"]), 3)

    def test_note_script_normalizes_record_paragraphs(self):
        module = self.load_note_script()
        paragraphs = module.normalize_record_paragraphs([
            "本次咨询开始后，咨询师说明保密和知情同意。\n\n来访者随后谈到工作压力。",
            "咨询师进一步追问具体情境，来访者讲述导师安排任务带来的焦虑。咨询师建议来访者保持冷静。",
        ])
        self.assertEqual(len(paragraphs), 3)
        self.assertTrue(paragraphs[-1].endswith("。"))
        self.assertNotIn("建议", paragraphs[-1])

    def test_parse_clinical_note_repairs_malformed_json(self):
        malformed_raw = (
            '```json { "session_summary": { "本次会谈整体摘要": "咨询师询问来访者今天的感觉，来访者表示还好" }, '
            '"soap": [ ["S (主观感觉)"]: ["烦躁", "焦虑"], ["O (客观表现)"]: [逐字稿中未明确提及], '
            '["A (评估分析)"]: [逐字稿中未明确提及], ["P (后续计划)"]: [逐字稿中未明确提及] ] } ```'
        )
        note = parse_clinical_note(malformed_raw)
        self.assertIsNone(note.raw_text)
        self.assertEqual(note.session_summary["本次会谈整体摘要"], ["咨询师询问来访者今天的感觉，来访者表示还好"])
        self.assertEqual(note.soap["S (主观感觉)"], ["烦躁", "焦虑"])
        self.assertEqual(note.soap["O (客观表现)"], ["逐字稿中未明确提及"])


if __name__ == "__main__":
    unittest.main()
