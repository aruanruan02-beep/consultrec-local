import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("config/local_settings.json")
ENV_PATH = Path(".env")


@dataclass
class LocalSettings:
    data_root: str = "data"
    whisper_command: str = ".venv/bin/python scripts/transcribe_mlx_whisper.py --audio {audio} --output {transcript_json} --model mlx-community/whisper-small-mlx --language zh"
    diarization_command: str = ".pyannote-venv/bin/python scripts/diarize_pyannote.py --audio {audio} --output {diarization_json} --num-speakers 2"
    llm_command: str = ""
    clinical_prompt_path: str = "prompts/clinical_note_prompt.md"
    hf_token: str = ""

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root)

    @property
    def prompt_path(self) -> Path:
        return Path(self.clinical_prompt_path)


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> LocalSettings:
    _load_dotenv(ENV_PATH)
    if not path.exists():
        return LocalSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = LocalSettings(**{**asdict(LocalSettings()), **data})
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or settings.hf_token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    return settings


def save_settings(settings: LocalSettings, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token


def settings_from_dict(data: Dict[str, Any]) -> LocalSettings:
    current = asdict(LocalSettings())
    for key in current:
        if key in data:
            current[key] = str(data[key])
    return LocalSettings(**current)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
