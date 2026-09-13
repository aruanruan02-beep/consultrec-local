#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROMPT = (
    "这是一段中文心理咨询对话。请忠实转写原话，保留嗯、停顿、重复和自我修正；"
    "使用自然的中文标点和完整句子。不要总结、解释、补充、猜测或添加说话人标签。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local timestamped ASR segments with MLX Whisper.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mlx-community/whisper-small-mlx")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--initial-prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    ensure_ffmpeg_on_path()
    import mlx.core as mx
    from mlx_whisper.load_models import load_model
    from mlx_whisper.transcribe import ModelHolder, transcribe

    total_started_at = time.perf_counter()
    model_started_at = time.perf_counter()
    print("Loading MLX Whisper transcription model...", flush=True)
    model = load_model(args.model, dtype=mx.float16)
    ModelHolder.model = model
    ModelHolder.model_path = args.model
    print(f"ASR model loaded in {time.perf_counter() - model_started_at:.3f} seconds.", flush=True)

    transcribe_started_at = time.perf_counter()
    print("Transcribing audio...", flush=True)
    result = transcribe(
        args.audio,
        path_or_hf_repo=args.model,
        language=args.language,
        verbose=None,
        initial_prompt=args.initial_prompt or None,
    )
    print(f"ASR transcription completed in {time.perf_counter() - transcribe_started_at:.3f} seconds.", flush=True)

    segments = normalize_mlx_segments(result.get("segments", []))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MLX Whisper completed in {time.perf_counter() - total_started_at:.3f} seconds.", flush=True)


def ensure_ffmpeg_on_path() -> None:
    import imageio_ffmpeg

    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = Path(tempfile.gettempdir()) / "consultrec-ffmpeg"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_link = ffmpeg_dir / "ffmpeg"
    if not ffmpeg_link.exists():
        ffmpeg_link.symlink_to(ffmpeg_path)
    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{ffmpeg_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"


def normalize_mlx_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []
    for item in segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        payload.append(
            {
                "start": float(item.get("start", 0)),
                "end": float(item.get("end", 0)),
                "speaker": "未确认",
                "text": text,
            }
        )
    return payload


if __name__ == "__main__":
    main()
