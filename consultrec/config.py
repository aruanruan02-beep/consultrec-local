import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("config/local_settings.json")


@dataclass
class LocalSettings:
    data_root: str = "data"
    whisper_command: str = ""
    diarization_command: str = ".venv/bin/python scripts/diarize_voice_features.py --audio {audio} --transcript-json {transcript_json} --output {diarization_json}"
    llm_command: str = ""
    clinical_prompt_path: str = "prompts/clinical_note_prompt.md"

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root)

    @property
    def prompt_path(self) -> Path:
        return Path(self.clinical_prompt_path)


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> LocalSettings:
    if not path.exists():
        return LocalSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    return LocalSettings(**{**asdict(LocalSettings()), **data})


def save_settings(settings: LocalSettings, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def settings_from_dict(data: Dict[str, Any]) -> LocalSettings:
    current = asdict(LocalSettings())
    for key in current:
        if key in data:
            current[key] = str(data[key])
    return LocalSettings(**current)
