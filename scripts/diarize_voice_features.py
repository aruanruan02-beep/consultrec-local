#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile


SPEAKER_LABELS = ["声音 A", "声音 B"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local lightweight two-speaker diarization.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    transcript = json.loads(Path(args.transcript_json).read_text(encoding="utf-8"))
    segments = transcript.get("segments", transcript)
    samples, sample_rate = decode_audio(Path(args.audio), args.sample_rate)
    features = np.array([segment_features(samples, sample_rate, item) for item in segments], dtype=np.float64)
    labels = cluster_two_speakers(features)

    output_segments = []
    for item, label in zip(segments, labels):
        output_segments.append(
            {
                "start": float(item.get("start", 0)),
                "end": float(item.get("end", 0)),
                "speaker": SPEAKER_LABELS[int(label)],
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"segments": output_segments}, ensure_ascii=False, indent=2), encoding="utf-8")


def decode_audio(audio_path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-vn",
            temp.name,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rate, data = wavfile.read(temp.name)
    samples = data.astype(np.float32)
    if samples.size and np.max(np.abs(samples)) > 0:
        samples = samples / np.max(np.abs(samples))
    return samples, rate


def segment_features(samples: np.ndarray, sample_rate: int, item: dict) -> np.ndarray:
    start = max(0, int(float(item.get("start", 0)) * sample_rate))
    end = min(len(samples), int(float(item.get("end", 0)) * sample_rate))
    chunk = samples[start:end]
    if chunk.size < sample_rate // 4:
        return np.zeros(18, dtype=np.float64)

    chunk = trim_silence(chunk)
    if chunk.size < sample_rate // 4:
        return np.zeros(18, dtype=np.float64)

    frame_size = int(sample_rate * 0.032)
    hop = int(sample_rate * 0.016)
    frames = frame_audio(chunk, frame_size, hop)
    if frames.size == 0:
        return np.zeros(18, dtype=np.float64)

    window = np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-8
    freqs = np.fft.rfftfreq(frame_size, d=1 / sample_rate)
    bands = band_energy(spectrum, freqs)
    log_bands = np.log(bands + 1e-8)

    centroid = (spectrum * freqs).sum(axis=1) / spectrum.sum(axis=1)
    rolloff = spectral_rolloff(spectrum, freqs)
    zcr = zero_crossing_rate(frames)
    energy = np.sqrt(np.mean(frames * frames, axis=1))

    return np.concatenate(
        [
            log_bands.mean(axis=0),
            log_bands.std(axis=0),
            [
                float(np.mean(centroid)),
                float(np.std(centroid)),
                float(np.mean(rolloff)),
                float(np.mean(zcr)),
                float(np.mean(energy)),
                float(np.std(energy)),
            ],
        ]
    )


def trim_silence(chunk: np.ndarray) -> np.ndarray:
    threshold = max(0.01, float(np.percentile(np.abs(chunk), 65)) * 0.4)
    indices = np.flatnonzero(np.abs(chunk) > threshold)
    if not indices.size:
        return chunk
    return chunk[indices[0] : indices[-1] + 1]


def frame_audio(samples: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if len(samples) < frame_size:
        samples = np.pad(samples, (0, frame_size - len(samples)))
    frame_count = 1 + max(0, (len(samples) - frame_size) // hop)
    return np.stack([samples[index * hop : index * hop + frame_size] for index in range(frame_count)])


def band_energy(spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    edges = np.array([80, 160, 260, 420, 680, 1100, 1800, 3000, 5000, 7600], dtype=np.float64)
    values = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        values.append(spectrum[:, mask].mean(axis=1) if mask.any() else np.zeros(spectrum.shape[0]))
    return np.stack(values, axis=1)


def spectral_rolloff(spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(spectrum, axis=1)
    thresholds = cumulative[:, -1] * 0.85
    indices = np.argmax(cumulative >= thresholds[:, None], axis=1)
    return freqs[indices]


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)


def cluster_two_speakers(features: np.ndarray) -> np.ndarray:
    if len(features) <= 1:
        return np.zeros(len(features), dtype=int)

    valid = np.any(np.abs(features) > 1e-8, axis=1)
    if valid.sum() <= 1:
        return np.zeros(len(features), dtype=int)

    normalized = features.copy()
    mean = normalized[valid].mean(axis=0)
    std = normalized[valid].std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (normalized - mean) / std

    valid_indices = np.flatnonzero(valid)
    valid_features = normalized[valid]
    first = valid_features[0]
    second = valid_features[np.argmax(np.linalg.norm(valid_features - first, axis=1))]
    centers = np.stack([first, second])

    labels = np.zeros(len(features), dtype=int)
    for _ in range(30):
        distances = np.linalg.norm(valid_features[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for group in range(2):
            members = valid_features[next_labels == group]
            if len(members):
                next_centers[group] = members.mean(axis=0)
        if np.array_equal(labels[valid_indices], next_labels):
            break
        labels[valid_indices] = next_labels
        centers = next_centers

    labels = smooth_labels(labels, valid)
    labels = normalize_label_order(labels)
    return labels


def smooth_labels(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    smoothed = labels.copy()
    for index in range(1, len(labels) - 1):
        if not valid[index]:
            smoothed[index] = smoothed[index - 1]
        elif labels[index - 1] == labels[index + 1] != labels[index]:
            smoothed[index] = labels[index - 1]
    return smoothed


def normalize_label_order(labels: np.ndarray) -> np.ndarray:
    if not len(labels):
        return labels
    first = int(labels[0])
    if first == 0:
        return labels
    return 1 - labels


if __name__ == "__main__":
    main()
