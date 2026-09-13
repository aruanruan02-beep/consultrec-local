#!/usr/bin/env python3
"""Run pyannote Community-1 locally and emit anonymous speaker turns."""

import argparse
import json
import os
import subprocess
import tempfile
import time
import resource
import wave
import warnings
from pathlib import Path


MODEL_ID = "pyannote/speaker-diarization-community-1"

# Audio is passed as a preloaded PCM waveform, so torchcodec is not used.
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local two-speaker pyannote diarization.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-speakers", type=int, default=2)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required for pyannote Community-1. Set it in the terminal environment and accept the model terms first."
        )

    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="consultrec-diarization-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        normalize_audio(Path(args.audio), wav_path)
        print("Loading local pyannote Community-1 model...", flush=True)
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
        if pipeline is None:
            raise RuntimeError(
                "Unable to load pyannote Community-1. Confirm the Hugging Face token and accept the model terms."
            )
        print(f"Diarization model loaded in {time.perf_counter() - started_at:.3f} seconds.", flush=True)
        inference_started_at = time.perf_counter()
        print("Detecting two local speaker timelines...", flush=True)
        output = pipeline(load_wav_waveform(wav_path), num_speakers=args.num_speakers)
        turns = getattr(output, "speaker_diarization", output)
        payload = []
        if hasattr(turns, "itertracks"):
            iterator = ((segment, speaker) for segment, _, speaker in turns.itertracks(yield_label=True))
        else:
            iterator = iter(turns)
        for segment, speaker in iterator:
            payload.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
        print(f"Diarization completed in {time.perf_counter() - inference_started_at:.3f} seconds.", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "turns": payload,
                "metrics": {
                    "model_load_seconds": inference_started_at - started_at,
                    "diarization_seconds": time.perf_counter() - inference_started_at,
                    "total_seconds": time.perf_counter() - started_at,
                    "peak_rss_bytes": _peak_rss_bytes(),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Pyannote diarization completed in {time.perf_counter() - started_at:.3f} seconds.", flush=True)


def normalize_audio(source: Path, output: Path) -> None:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
    ]
    subprocess.run(command, check=True)


def load_wav_waveform(path: Path):
    import numpy as np
    import torch

    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise RuntimeError("Expected normalized 16-bit PCM WAV audio.")
        frames = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        waveform = frames.reshape(-1, reader.getnchannels()).T.astype(np.float32) / 32768.0
        sample_rate = reader.getframerate()
    return {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(value if value > 10_000_000 else value * 1024)


if __name__ == "__main__":
    main()
