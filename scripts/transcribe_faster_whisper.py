#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe local audio with faster-whisper.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments, _info = model.transcribe(
        args.audio,
        language=args.language,
        vad_filter=True,
    )

    payload = []
    for item in segments:
        payload.append(
            {
                "start": float(item.start),
                "end": float(item.end),
                "speaker": "Unknown",
                "text": item.text.strip(),
            }
        )

    output.write_text(json.dumps({"segments": payload}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

