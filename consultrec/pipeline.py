from pathlib import Path
from typing import Optional

from .asr import run_asr_command
from .diarization import assign_roles
from .llm import run_llm_command
from .prompts import load_clinical_prompt
from .render import write_outputs
from .transcript import load_transcript_json, save_transcript_json, transcript_for_prompt


def process_session(
    audio_path: Path,
    output_dir: Path,
    session_id: str,
    prompt_path: Path,
    llm_command: str,
    asr_command: Optional[str] = None,
    transcript_json: Optional[Path] = None,
    roles_mode: str = "existing",
    diarization_command: Optional[str] = None,
    therapist_speaker: Optional[str] = None,
    client_speaker: Optional[str] = None,
) -> None:
    if transcript_json:
        segments = load_transcript_json(transcript_json)
    elif asr_command:
        segments = run_asr_command(asr_command, audio_path, output_dir / ".work" / session_id / "asr")
    else:
        raise ValueError("Provide either --transcript-json or --asr-command.")

    segments = assign_roles(
        segments=segments,
        mode=roles_mode,
        therapist_speaker=therapist_speaker,
        client_speaker=client_speaker,
        diarization_command=diarization_command,
        audio_path=audio_path,
        work_dir=output_dir / ".work" / session_id / "diarization",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_transcript_json(output_dir / f"{session_id}.transcript.json", segments)

    prompt = load_clinical_prompt(prompt_path, transcript_for_prompt(segments))
    note = run_llm_command(llm_command, prompt, output_dir / ".work" / session_id / "llm")
    write_outputs(output_dir, session_id, audio_path, segments, note)

