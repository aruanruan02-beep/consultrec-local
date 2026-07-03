#!/usr/bin/env python3
# Limit thread counts for CPU libraries to prevent 100% CPU starvation on M1 Mac
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


UNKNOWN_SPEAKER = "未确认"


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe and diarize local audio with WhisperX.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "WhisperX diarization requires HF_TOKEN or HUGGINGFACE_TOKEN. "
            "Create a Hugging Face read token and accept the pyannote diarization model terms first."
        )

    segments = transcribe_with_whisperx(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
        hf_token=token,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def transcribe_with_whisperx(
    audio_path: str,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    min_speakers: int,
    max_speakers: int,
    batch_size: int,
    hf_token: str,
) -> List[Dict[str, Any]]:
    ensure_ffmpeg_on_path()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import gc
    import torch
    torch.set_num_threads(4)
    import whisperx

    # 1. Transcribe (runs on CPU by default as CTranslate2 lacks MPS support)
    print("Loading Whisper transcription model...")
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    print("Transcribing audio...")
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # Free transcription model memory immediately
    print("Freeing transcription model memory...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 2. Align (runs on MPS for GPU acceleration if available)
    language_code = result.get("language") or language
    align_device = "mps" if (device == "cpu" and torch.backends.mps.is_available()) else device

    print(f"Loading alignment model on device: {align_device}...")
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=align_device)
        print("Aligning segments...")
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            align_device,
            return_char_alignments=False,
        )
    except Exception as e:
        if align_device == "mps":
            print(f"MPS alignment failed ({e}), falling back to CPU...")
            align_device = "cpu"
            align_model, metadata = whisperx.load_align_model(language_code=language_code, device=align_device)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                align_device,
                return_char_alignments=False,
            )
        else:
            raise e

    # Free alignment model memory immediately
    print("Freeing alignment model memory...")
    del align_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 3. Diarize (runs on MPS for GPU acceleration if available)
    diarize_device = "mps" if (device == "cpu" and torch.backends.mps.is_available()) else device

    print(f"Loading diarization pipeline on device: {diarize_device}...")
    try:
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=diarize_device)
        print("Diarizing speakers...")
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as e:
        if diarize_device == "mps":
            print(f"MPS diarization failed ({e}), falling back to CPU...")
            diarize_device = "cpu"
            diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=diarize_device)
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        else:
            raise e

    # Free diarization model memory immediately
    print("Freeing diarization model memory...")
    del diarize_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Final speaker assignment
    print("Assigning speakers to segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return normalize_whisperx_segments(result.get("segments", []))


def ensure_ffmpeg_on_path() -> None:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        return

    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = Path(tempfile.gettempdir()) / "consultrec-ffmpeg"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_link = ffmpeg_dir / "ffmpeg"
    if not ffmpeg_link.exists():
        try:
            ffmpeg_link.symlink_to(ffmpeg_path)
        except FileExistsError:
            pass
    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{ffmpeg_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"


def normalize_whisperx_segments(segments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    speaker_map: Dict[str, str] = {}
    payload: List[Dict[str, Any]] = []
    try:
        import zhconv
    except ImportError:
        zhconv = None

    for item in segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if zhconv:
            text = zhconv.convert(text, 'zh-hans')
        speaker = normalize_speaker_label(item.get("speaker"), speaker_map)
        payload.append(
            {
                "start": float(item.get("start", 0)),
                "end": float(item.get("end", 0)),
                "speaker": speaker,
                "text": text,
            }
        )
    return payload
def normalize_speaker_label(raw_speaker: object, speaker_map: Dict[str, str]) -> str:
    speaker = str(raw_speaker or "").strip()
    if not speaker:
        return UNKNOWN_SPEAKER
    if speaker not in speaker_map:
        speaker_map[speaker] = f"说话人 {len(speaker_map) + 1}"
    return speaker_map[speaker]


if __name__ == "__main__":
    main()
