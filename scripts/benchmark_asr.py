#!/usr/bin/env python3
"""Run a local ASR command and persist comparable benchmark metrics.

The command is supplied by the caller so this script does not make MLX a
runtime dependency of the application. Use the same audio and output schema
for each candidate engine.
"""

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a local ASR command.")
    parser.add_argument("--label", required=True, help="Engine/model label, for example mlx-whisper-small")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True, help="Transcript JSON created by the ASR command")
    parser.add_argument("--report", required=True, help="Benchmark report JSON")
    parser.add_argument(
        "--command",
        required=True,
        help="Command template containing {audio} and {output} placeholders.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    command = args.command.format(audio=shlex.quote(args.audio), output=shlex.quote(str(output_path)))
    started_at = time.perf_counter()
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak_rss_mb = 0.0
    peak_swap_mb = _swap_used_mb()
    logs = []

    while process.poll() is None:
        peak_rss_mb = max(peak_rss_mb, _rss_mb(process.pid))
        current_swap_mb = _swap_used_mb()
        if current_swap_mb is not None:
            peak_swap_mb = max(peak_swap_mb or 0.0, current_swap_mb)
        time.sleep(0.25)

    if process.stdout:
        logs = process.stdout.read().splitlines()
    peak_rss_mb = max(peak_rss_mb, _rss_mb(process.pid))
    if process.returncode:
        raise SystemExit("\n".join(logs) or f"ASR command failed with exit code {process.returncode}.")
    if not output_path.exists():
        raise SystemExit("ASR command completed without producing its output JSON.")

    transcript = json.loads(output_path.read_text(encoding="utf-8"))
    segments = transcript.get("segments", [])
    duration_seconds = _audio_duration_seconds(args.audio)
    total_seconds = time.perf_counter() - started_at
    report: Dict[str, Any] = {
        "label": args.label,
        "audio": str(Path(args.audio).resolve()),
        "audio_duration_seconds": duration_seconds,
        "model_load_seconds": _seconds_from_logs(logs, r"ASR model loaded in ([0-9.]+) seconds\\.$"),
        "transcription_seconds": _seconds_from_logs(logs, r"ASR transcription completed in ([0-9.]+) seconds\\.$"),
        "total_seconds": round(total_seconds, 3),
        "realtime_factor": round(total_seconds / duration_seconds, 3) if duration_seconds else None,
        "peak_rss_mb": round(peak_rss_mb, 1) if peak_rss_mb else None,
        "peak_system_swap_mb": round(peak_swap_mb, 1) if peak_swap_mb is not None else None,
        "segment_count": len(segments) if isinstance(segments, list) else None,
        "quality_review": {
            "whole_sentence_omissions": None,
            "chinese_character_errors": None,
            "filler_word_retention": None,
            "segment_timestamp_usability": None,
            "notes": "Fill these fields after human review of the same reference audio.",
        },
        "logs": logs,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _rss_mb(pid: int) -> float:
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True)
        return int(result.stdout.strip() or "0") / 1024
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0.0


def _swap_used_mb() -> Optional[float]:
    try:
        result = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True)
        match = re.search(r"used = ([0-9.]+)M", result.stdout)
        return float(match.group(1)) if match else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _audio_duration_seconds(audio_path: str) -> Optional[float]:
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", audio_path], capture_output=True, text=True, check=True)
        return round(float(result.stdout.strip()), 3)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _seconds_from_logs(lines: list[str], pattern: str) -> Optional[float]:
    for line in lines:
        match = re.search(pattern, line)
        if match:
            return float(match.group(1))
    return None


if __name__ == "__main__":
    main()
