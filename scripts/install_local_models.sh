#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "Python dependencies installed."
echo ""
echo "WhisperX diarization requires a Hugging Face read token."
echo "Accept the pyannote speaker diarization model terms, then run:"
echo ""
echo "  export HF_TOKEN=\"your_huggingface_read_token\""
echo ""
echo "Next install Ollama from https://ollama.com/download/mac if it is not installed."
echo "After Ollama is running, run:"
echo ""
echo "  ollama pull qwen2.5:7b-instruct"
echo ""
echo "Then update config/local_settings.json with:"
echo ""
echo '  "whisper_command": ".venv/bin/python scripts/transcribe_whisperx.py --audio {audio} --output {transcript_json} --model small --language zh --device cpu --compute-type int8 --min-speakers 2 --max-speakers 2",'
echo '  "diarization_command": "",'
echo '  "llm_command": ".venv/bin/python scripts/generate_note_ollama.py --prompt-file {prompt_file} --model qwen2.5:7b-instruct"'
echo ""
