from __future__ import annotations

"""Build a raw-window DB5 dataset for deep learning experiments.

Examples
--------
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 200 --step-ms 50

python db5_windows.py --subjects 1-10 --gestures 1-10 \
    --out data/ninapro_db5/windows/db5_s1-10_g1-10_win200_step50.npz

Window-size experiments
-----------------------
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 200 --step-ms 50
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 300 --step-ms 50
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 400 --step-ms 50
"""

import argparse
import json
import os
import re
from typing import Dict, List, Sequence

import numpy as np
from scipy.io import loadmat

from src.ninapro_db5 import RAW_DIR, unzip_subject


DEFAULT_SUBJECTS = [1, 2, 3]
DEFAULT_GESTURES = list(range(1, 11))
DEFAULT_EXERCISES = [1, 2, 3]
MAT_FILE_PATTERN = re.compile(r"^S(?P<subject>\d+)_E(?P<exercise>\d+)_A1\.mat$")


def parse_int_list(text: str | Sequence[int]) -> List[int]:
    if isinstance(text, (list, tuple, np.ndarray)):
        values: List[int] = []
        seen: set[int] = set()
        for item in text:
            number = int(item)
            if number not in seen:
                values.append(number)
                seen.add(number)
        return values

    stripped = str(text).strip()
    if not stripped:
        return []

    values = []
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


def format_range(values: Sequence[int], prefix: str) -> str:
    ordered = list(values)
    if not ordered:
        return f"{prefix}none"
    if len(ordered) >= 2 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{prefix}{ordered[0]}-{ordered[-1]}"
    return prefix + "_".join(str(value) for value in ordered)


def default_output_path(subjects: Sequence[int], gestures: Sequence[int], win_sec: float, step_sec: float) -> str:
    subject_tag = format_range(subjects, "s")
    gesture_tag = format_range(gestures, "g")
    win_ms = int(round(win_sec * 1000))
    step_ms = int(round(step_sec * 1000))
    file_name = f"db5_{subject_tag}_{gesture_tag}_win{win_ms}_step{step_ms}.npz"
    return os.path.join("data", "ninapro_db5", "windows", file_name)


def resolve_window_step_seconds(
    win_sec: float,
    step_sec: float,
    window_ms: float | None,
    step_ms: float | None,
) -> tuple[float, float]:
    resolved_win_sec = float(win_sec)
    resolved_step_sec = float(step_sec)

    if window_ms is not None:
        resolved_win_sec = float(window_ms) / 1000.0
    if step_ms is not None:
        resolved_step_sec = float(step_ms) / 1000.0

    if resolved_win_sec <= 0 or resolved_step_sec <= 0:
        raise ValueError("Window and step length must be positive.")

    return resolved_win_sec, resolved_step_sec


def find_mat_path(raw_root: str, subject_id: int, exercise_id: int) -> str | None:
    file_name = f"S{subject_id}_E{exercise_id}_A1.mat"
    candidate_paths = [
        os.path.join(raw_root, f"s{subject_id}", file_name),
        os.path.join(raw_root, f"s{subject_id}", f"s{subject_id}", file_name),
    ]

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    search_root = os.path.join(raw_root, f"s{subject_id}")
    if not os.path.isdir(search_root):
        return None

    matches: List[str] = []
    for current_root, _dir_names, file_names in os.walk(search_root):
        if file_name in file_names:
            matches.append(os.path.join(current_root, file_name))
    if not matches:
        return None
    return sorted(matches)[0]


def parse_mat_identity(mat_path: str) -> tuple[int, int]:
    file_name = os.path.basename(mat_path)
    match = MAT_FILE_PATTERN.fullmatch(file_name)
    if match is None:
        raise ValueError(
            f"Unexpected DB5 file name format: {file_name}. Expected S<subject>_E<exercise>_A1.mat"
        )
    return int(match.group("subject")), int(match.group("exercise"))


def build_db5_window_dataset(
    subjects: Sequence[int],
    gestures: Sequence[int],
    exercises: Sequence[int],
    max_reps: int,
    win_sec: float,
    step_sec: float,
    raw_root: str = RAW_DIR,
) -> Dict[str, np.ndarray]:
    requested_subjects = sorted({int(subject_id) for subject_id in subjects})
    class_gesture_ids = [int(gesture_id) for gesture_id in gestures]
    gesture_to_class = {gesture_id: class_index for class_index, gesture_id in enumerate(class_gesture_ids)}

    windows: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[int] = []
    exercise_ids: List[int] = []
    gesture_ids: List[int] = []
    repetition_ids: List[int] = []
    window_starts: List[int] = []
    window_ends: List[int] = []

    sampling_rate_hz: int | None = None
    window_size_samples: int | None = None
    step_size_samples: int | None = None
    num_channels: int | None = None

    for subject_id in subjects:
        unzip_subject(int(subject_id))

        for exercise_id in exercises:
            mat_path = find_mat_path(raw_root=raw_root, subject_id=int(subject_id), exercise_id=int(exercise_id))
            if mat_path is None:
                print(f"[WARN] Missing file for subject {subject_id}, exercise {exercise_id}. Skip.")
                continue

            print(f"[INFO] Loading {mat_path}")
            parsed_subject_id, parsed_exercise_id = parse_mat_identity(mat_path)
            if parsed_subject_id != int(subject_id) or parsed_exercise_id != int(exercise_id):
                raise AssertionError(
                    "DB5 file identity mismatch: "
                    f"loop subject/exercise=({int(subject_id)}, {int(exercise_id)}), "
                    f"file parsed subject/exercise=({parsed_subject_id}, {parsed_exercise_id})"
                )

            mat = loadmat(mat_path)

            emg = np.asarray(mat["emg"], dtype=np.float32)
            restimulus = np.asarray(mat["restimulus"]).ravel().astype(np.int64)
            rerepetition = np.asarray(mat["rerepetition"]).ravel().astype(np.int64)
            file_sampling_rate = int(np.asarray(mat.get("frequency", np.asarray([200]))).ravel()[0])
            file_subject_id = int(subject_id)
            file_exercise_id = int(exercise_id)

            file_window_size = int(round(win_sec * file_sampling_rate))
            file_step_size = int(round(step_sec * file_sampling_rate))
            if file_window_size <= 0 or file_step_size <= 0:
                raise ValueError("Window length and step length must be positive.")

            if sampling_rate_hz is None:
                sampling_rate_hz = file_sampling_rate
                window_size_samples = file_window_size
                step_size_samples = file_step_size
                num_channels = int(emg.shape[1])
            else:
                if file_sampling_rate != sampling_rate_hz:
                    raise ValueError(
                        f"Inconsistent sampling rate: expected {sampling_rate_hz}, got {file_sampling_rate}"
                    )
                if file_window_size != window_size_samples:
                    raise ValueError(
                        f"Inconsistent window size: expected {window_size_samples}, got {file_window_size}"
                    )
                if file_step_size != step_size_samples:
                    raise ValueError(
                        f"Inconsistent step size: expected {step_size_samples}, got {file_step_size}"
                    )
                if int(emg.shape[1]) != num_channels:
                    raise ValueError(f"Inconsistent channel count: expected {num_channels}, got {emg.shape[1]}")

            file_window_count = 0
            for gesture_id in class_gesture_ids:
                for repetition_id in range(1, int(max_reps) + 1):
                    mask = (restimulus == gesture_id) & (rerepetition == repetition_id)
                    indices = np.where(mask)[0]
                    if indices.size == 0:
                        continue

                    span_start = int(indices[0])
                    span_end = int(indices[-1]) + 1
                    if span_end - span_start < file_window_size:
                        continue

                    for window_start in range(span_start, span_end - file_window_size + 1, file_step_size):
                        window_end = window_start + file_window_size
                        windows.append(emg[window_start:window_end, :].astype(np.float32, copy=False))
                        labels.append(gesture_to_class[gesture_id])
                        subject_ids.append(file_subject_id)
                        exercise_ids.append(file_exercise_id)
                        gesture_ids.append(gesture_id)
                        repetition_ids.append(repetition_id)
                        window_starts.append(window_start)
                        window_ends.append(window_end)
                        file_window_count += 1

            print(f"[INFO] Extracted {file_window_count} windows from subject {subject_id}, exercise {exercise_id}")

    if not windows:
        raise RuntimeError("No windows were extracted. Check zip files, gestures, exercises, and repetitions.")

    x = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    dataset = {
        "x": x,
        "y": y,
        "subject_id": np.asarray(subject_ids, dtype=np.int64),
        "exercise_id": np.asarray(exercise_ids, dtype=np.int64),
        "gesture_id": np.asarray(gesture_ids, dtype=np.int64),
        "repetition_id": np.asarray(repetition_ids, dtype=np.int64),
        "window_start": np.asarray(window_starts, dtype=np.int64),
        "window_end": np.asarray(window_ends, dtype=np.int64),
        "class_gesture_ids": np.asarray(class_gesture_ids, dtype=np.int64),
        "sampling_rate_hz": np.asarray(int(sampling_rate_hz), dtype=np.int64),
        "window_size_samples": np.asarray(int(window_size_samples), dtype=np.int64),
        "step_size_samples": np.asarray(int(step_size_samples), dtype=np.int64),
        "num_channels": np.asarray(int(num_channels), dtype=np.int64),
    }

    discovered_subjects = sorted(np.unique(dataset["subject_id"]).astype(np.int64).tolist())
    print(f"[INFO] Requested subjects: {requested_subjects}")
    print(f"[INFO] Discovered subjects in saved metadata: {discovered_subjects}")
    if discovered_subjects != requested_subjects:
        raise AssertionError(
            f"Saved metadata subject_id set {discovered_subjects} does not match requested subjects {requested_subjects}"
        )

    return dataset


def dataset_summary(dataset: Dict[str, np.ndarray]) -> Dict[str, object]:
    unique_subject_ids = [int(value) for value in np.unique(dataset["subject_id"]).tolist()]
    summary = {
        "num_windows": int(dataset["x"].shape[0]),
        "window_shape": [int(dataset["x"].shape[1]), int(dataset["x"].shape[2])],
        "num_classes": int(len(dataset["class_gesture_ids"])),
        "class_gesture_ids": [int(value) for value in dataset["class_gesture_ids"].tolist()],
        "subjects": unique_subject_ids,
        "unique_subject_ids": unique_subject_ids,
        "exercises": [int(value) for value in np.unique(dataset["exercise_id"]).tolist()],
        "repetitions": [int(value) for value in np.unique(dataset["repetition_id"]).tolist()],
        "sampling_rate_hz": int(np.asarray(dataset["sampling_rate_hz"]).item()),
        "window_size_samples": int(np.asarray(dataset["window_size_samples"]).item()),
        "step_size_samples": int(np.asarray(dataset["step_size_samples"]).item()),
    }
    return summary


def save_dataset(dataset: Dict[str, np.ndarray], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(output_path, **dataset)

    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(dataset_summary(dataset), summary_file, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved dataset to {output_path}")
    print(f"[INFO] Saved summary to {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build raw EMG window datasets from NinaPro DB5 .mat files.")
    parser.add_argument("--subjects", type=str, default="1-3", help="Subject IDs, for example 1-3 or 1,2,3")
    parser.add_argument("--gestures", type=str, default="1-10", help="Gesture IDs, for example 1-10")
    parser.add_argument("--exercises", type=str, default="1-3", help="Exercise IDs, for example 1-3")
    parser.add_argument("--max-reps", type=int, default=6, help="Maximum repetition ID to scan")
    parser.add_argument("--win-sec", type=float, default=0.200, help="Window length in seconds")
    parser.add_argument("--step-sec", type=float, default=0.050, help="Window step in seconds")
    parser.add_argument("--window-ms", type=float, default=None, help="Window length in milliseconds, overrides --win-sec")
    parser.add_argument("--step-ms", type=float, default=None, help="Window step in milliseconds, overrides --step-sec")
    parser.add_argument("--raw-dir", type=str, default=RAW_DIR, help="DB5 raw directory root")
    parser.add_argument("--out", type=str, default="", help="Output .npz path")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    subjects = parse_int_list(args.subjects) or DEFAULT_SUBJECTS
    gestures = parse_int_list(args.gestures) or DEFAULT_GESTURES
    exercises = parse_int_list(args.exercises) or DEFAULT_EXERCISES
    resolved_win_sec, resolved_step_sec = resolve_window_step_seconds(
        win_sec=args.win_sec,
        step_sec=args.step_sec,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
    )
    output_path = args.out or default_output_path(subjects, gestures, resolved_win_sec, resolved_step_sec)

    print(f"[INFO] Window configuration: window_ms={resolved_win_sec * 1000:.0f}, step_ms={resolved_step_sec * 1000:.0f}")

    dataset = build_db5_window_dataset(
        subjects=subjects,
        gestures=gestures,
        exercises=exercises,
        max_reps=args.max_reps,
        win_sec=resolved_win_sec,
        step_sec=resolved_step_sec,
        raw_root=args.raw_dir,
    )
    save_dataset(dataset, output_path)
    print(json.dumps(dataset_summary(dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
