import shlex
import subprocess
from pathlib import Path
from typing import List

from .models import TranscriptSegment
from .transcript import load_transcript_json


def run_asr_command(command_template: str, audio_path: Path, work_dir: Path) -> List[TranscriptSegment]:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_stem = work_dir / audio_path.stem
    command = command_template.format(
        audio=str(audio_path),
        asr_output_stem=str(output_stem),
        asr_output_json=str(output_stem.with_suffix(".json")),
    )
    subprocess.run(shlex.split(command), check=True)
    output_json = output_stem.with_suffix(".json")
    if not output_json.exists():
        raise FileNotFoundError(
            "ASR command completed but did not create JSON output at "
            f"{output_json}. Use {{asr_output_stem}} or {{asr_output_json}} in the command."
        )
    return load_transcript_json(output_json)

