#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import ollama

OLLAMA_CANDIDATES = [
    Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    Path("/usr/local/bin/ollama"),
    Path("/opt/homebrew/bin/ollama"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clinical note with a local Ollama model.")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    args = parser.parse_args()

    prompt = open(args.prompt_file, "r", encoding="utf-8").read()
    ensure_ollama_running()
    response = ollama.generate(
        model=args.model,
        prompt=prompt,
        options={
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    )
    sys.stdout.write(response.get("response", ""))


def ensure_ollama_running() -> None:
    if can_reach_ollama():
        return
    executable = next((path for path in OLLAMA_CANDIDATES if path.exists()), None)
    if not executable:
        return
    subprocess.Popen(
        [str(executable), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        if can_reach_ollama():
            return
        time.sleep(0.5)


def can_reach_ollama() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1):
            return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
