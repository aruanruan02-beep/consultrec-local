from pathlib import Path


def load_clinical_prompt(prompt_path: Path, transcript_text: str) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    if "{{TRANSCRIPT}}" not in template:
        raise ValueError("Prompt file must include the {{TRANSCRIPT}} placeholder.")
    return template.replace("{{TRANSCRIPT}}", transcript_text)

