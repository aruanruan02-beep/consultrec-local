from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker or "未确认",
            "text": self.text,
        }


@dataclass
class ClinicalNote:
    soap: Dict[str, List[str]]
    session_summary: Dict[str, List[str]]
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "soap": self.soap,
            "session_summary": self.session_summary,
        }
        if self.raw_text:
            data["raw_text"] = self.raw_text
        return data
