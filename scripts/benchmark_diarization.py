#!/usr/bin/env python3
"""Score local diarization turns against the manually corrected transcript."""

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROLE_PATTERN = re.compile(r"^(咨询师|来访者)\s+(?:(\d+):)?(\d{2}):(\d{2})\s*$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark diarization turns using a corrected transcript.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--diarization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()

    reference = limit_window(load_reference(Path(args.reference)), args.start, args.end)
    payload = json.loads(Path(args.diarization).read_text(encoding="utf-8"))
    turns = limit_window(
        sorted(payload.get("turns", []), key=lambda item: (float(item["start"]), float(item["end"]))), args.start, args.end
    )
    report = {
        "reference_turns": len(reference),
        "predicted_turns": len(turns),
        "boundary_metrics": {
            "one_second": boundary_metrics(reference, turns, 1.0),
            "two_seconds": boundary_metrics(reference, turns, 2.0),
        },
        "speaker_accuracy": speaker_accuracy(reference, turns),
        "runtime": payload.get("metrics", {}),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_reference(path: Path) -> List[Dict[str, object]]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROLE_PATTERN.match(line.strip())
        if not match:
            continue
        hours = int(match.group(2) or 0)
        start = hours * 3600 + int(match.group(3)) * 60 + int(match.group(4))
        turns.append({"speaker": match.group(1), "start": float(start)})
    for index, turn in enumerate(turns):
        turn["end"] = turns[index + 1]["start"] if index + 1 < len(turns) else turn["start"] + 1.0
    return turns


def limit_window(turns: List[Dict[str, object]], start: float, end: Optional[float]) -> List[Dict[str, object]]:
    limited = []
    for turn in turns:
        turn_start = float(turn["start"])
        turn_end = float(turn["end"])
        if turn_end <= start or (end is not None and turn_start >= end):
            continue
        item = dict(turn)
        item["start"] = max(turn_start, start)
        item["end"] = min(turn_end, end) if end is not None else turn_end
        limited.append(item)
    return limited


def boundary_metrics(reference: List[Dict[str, object]], predicted: List[Dict[str, object]], tolerance: float) -> Dict[str, float]:
    expected = [float(item["start"]) for item in reference[1:]]
    actual = [float(item["start"]) for item in predicted[1:]]
    matched_actual = set()
    hits = 0
    for expected_boundary in expected:
        candidates = [
            (abs(actual_boundary - expected_boundary), index)
            for index, actual_boundary in enumerate(actual)
            if index not in matched_actual and abs(actual_boundary - expected_boundary) <= tolerance
        ]
        if candidates:
            _, index = min(candidates)
            matched_actual.add(index)
            hits += 1
    precision = hits / len(actual) if actual else 0.0
    recall = hits / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def speaker_accuracy(reference: List[Dict[str, object]], predicted: List[Dict[str, object]]) -> Dict[str, object]:
    predicted = [item for item in predicted if str(item.get("speaker", "")) not in {"", "未确认", "Unknown"}]
    labels = sorted({str(item.get("speaker", "")) for item in predicted})
    if len(labels) != 2:
        return {"accuracy": None, "mapping": {}, "reason": "Expected exactly two predicted speaker labels."}
    roles = ["咨询师", "来访者"]
    best = (0.0, {})
    for mapped_roles in itertools.permutations(roles):
        mapping = dict(zip(labels, mapped_roles))
        correct = total = 0.0
        for predicted_turn in predicted:
            for reference_turn in reference:
                overlap = max(
                    0.0,
                    min(float(predicted_turn["end"]), float(reference_turn["end"]))
                    - max(float(predicted_turn["start"]), float(reference_turn["start"])),
                )
                total += overlap
                if overlap and mapping[str(predicted_turn["speaker"])] == reference_turn["speaker"]:
                    correct += overlap
        if total and correct / total > best[0]:
            best = (correct / total, mapping)
    return {"accuracy": round(best[0], 4), "mapping": best[1]}


if __name__ == "__main__":
    main()
