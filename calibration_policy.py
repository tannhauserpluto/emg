from __future__ import annotations

"""Decide whether to run calibration based on baseline evaluation results."""

import argparse
import csv
import json
import os
from typing import Iterable, Sequence


def parse_int_list(text: str) -> list[int]:
    stripped = str(text).strip()
    if not stripped:
        return []
    values: list[int] = []
    seen: set[int] = set()
    for chunk in stripped.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid range: {token}")
            numbers = range(start, end + 1)
        else:
            numbers = [int(token)]
        for number in numbers:
            if number not in seen:
                values.append(number)
                seen.add(number)
    return values


def load_metrics(metrics_path: str) -> dict[str, float]:
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    if "accuracy" in metrics and "macro_f1" in metrics:
        return {"accuracy": float(metrics["accuracy"]), "macro_f1": float(metrics["macro_f1"])}
    if "all" in metrics and "focus" in metrics:
        raise ValueError("Calibration policy expects a baseline metrics file, not a before/after calibration summary.")
    raise ValueError(f"Metrics file {metrics_path} must include accuracy and macro_f1.")


def load_per_class_accuracy(file_path: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with open(file_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Per-class accuracy file {file_path} is missing a header row.")
        required_fields = {"gesture_id", "accuracy"}
        missing_fields = required_fields - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                f"Per-class accuracy file {file_path} is missing required columns: {sorted(missing_fields)}"
            )

        valid_rows: list[dict[str, object]] = []
        ignored_rows: list[dict[str, object]] = []
        for row in reader:
            gesture_id = int(row["gesture_id"])
            accuracy = float(row["accuracy"])
            support = int(row.get("support", 0))
            record = {"gesture_id": gesture_id, "accuracy": accuracy, "support": support}
            if support <= 0:
                ignored_rows.append(record)
            else:
                valid_rows.append(record)

    if not valid_rows:
        raise ValueError(f"Per-class accuracy file {file_path} contains no valid rows with support > 0.")

    valid_rows.sort(key=lambda item: (float(item["accuracy"]), int(item["support"])))
    ignored_rows.sort(key=lambda item: (int(item["gesture_id"]), int(item["support"])))
    return valid_rows, ignored_rows


def select_worst_gestures(
    ranking: list[dict[str, object]],
    worst_k: int,
    min_accuracy: float | None,
) -> list[int]:
    candidates = ranking
    if min_accuracy is not None:
        filtered = [row for row in ranking if float(row["accuracy"]) <= float(min_accuracy)]
        if filtered:
            candidates = filtered
        else:
            print(
                f"[INFO] No gestures at or below accuracy {min_accuracy:.4f}; falling back to worst-{worst_k}."
            )

    worst_k = int(worst_k)
    if worst_k <= 0:
        raise ValueError(f"auto-select worst_k must be positive, got {worst_k}")
    worst_k = min(worst_k, len(candidates))
    return [int(row["gesture_id"]) for row in candidates[:worst_k]]


def filter_ranking_by_gestures(
    ranking: list[dict[str, object]],
    gesture_ids: Iterable[int],
) -> list[dict[str, object]]:
    target_set = {int(value) for value in gesture_ids}
    return [row for row in ranking if int(row["gesture_id"]) in target_set]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide whether to run calibration for a target subject.")
    parser.add_argument("--metrics-json", type=str, required=True, help="Baseline metrics JSON file")
    parser.add_argument("--per-class-csv", type=str, required=True, help="Baseline per-class accuracy CSV file")
    parser.add_argument("--target-subject", type=int, default=None, help="Optional target subject ID for logging")
    parser.add_argument("--important-gestures", type=str, default="", help="Comma-separated list of important gestures")
    parser.add_argument("--auto-select-worst-k", type=int, default=3, help="Worst-K gestures to recommend when calibrating")
    parser.add_argument("--calibration-trigger-all-acc", type=float, default=0.65)
    parser.add_argument("--calibration-trigger-class-acc", type=float, default=0.50)
    parser.add_argument("--min-accuracy", type=float, default=None, help="Optional minimum accuracy threshold for selection")
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_json)
    ranking, ignored = load_per_class_accuracy(args.per_class_csv)
    important_gestures = parse_int_list(args.important_gestures)

    subject_label = f"subject {int(args.target_subject)}" if args.target_subject is not None else "target subject"
    print(f"[INFO] Baseline summary for {subject_label}:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("[INFO] Valid per-class ranking (support > 0 only):")
    print(json.dumps(ranking, ensure_ascii=False, indent=2))
    if ignored:
        print("[INFO] Ignored gestures with support <= 0:")
        print(json.dumps(ignored, ensure_ascii=False, indent=2))

    overall_trigger = float(metrics["accuracy"]) < float(args.calibration_trigger_all_acc)

    if important_gestures:
        important_ranking = filter_ranking_by_gestures(ranking, important_gestures)
    else:
        important_ranking = ranking

    below_threshold = [
        row
        for row in important_ranking
        if float(row["accuracy"]) < float(args.calibration_trigger_class_acc)
    ]
    class_trigger = bool(below_threshold)

    should_calibrate = overall_trigger or class_trigger
    if should_calibrate:
        if below_threshold:
            gesture_candidates = below_threshold
            reason = "important gesture(s) below class accuracy threshold"
        else:
            gesture_candidates = ranking
            reason = "overall accuracy below threshold"

        suggested = select_worst_gestures(
            ranking=gesture_candidates,
            worst_k=int(args.auto_select_worst_k),
            min_accuracy=args.min_accuracy,
        )
        print(f"[INFO] Recommended action: calibrate ({reason}).")
        print(f"[INFO] Suggested target gestures: {suggested}")
    else:
        print(
            "[INFO] Recommended action: skip calibration. Baseline accuracy is strong and no important gestures "
            "fall below the class threshold; calibration risk may outweigh benefit."
        )
        suggested = []

    summary = {
        "target_subject": int(args.target_subject) if args.target_subject is not None else None,
        "metrics": metrics,
        "overall_trigger": bool(overall_trigger),
        "class_trigger": bool(class_trigger),
        "important_gestures": [int(value) for value in important_gestures],
        "thresholds": {
            "overall_accuracy": float(args.calibration_trigger_all_acc),
            "class_accuracy": float(args.calibration_trigger_class_acc),
            "min_accuracy": float(args.min_accuracy) if args.min_accuracy is not None else None,
        },
        "recommended_action": "calibrate" if should_calibrate else "skip",
        "suggested_gestures": suggested,
    }

    if args.out:
        output_dir = os.path.dirname(args.out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved calibration policy decision to {args.out}")


if __name__ == "__main__":
    main()
