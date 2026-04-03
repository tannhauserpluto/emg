from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Sequence

import numpy as np


ROW_METADATA_KEYS = (
    "subject_id",
    "exercise_id",
    "gesture_id",
    "repetition_id",
    "window_start",
    "window_end",
)


def row_key_from_dataset(data: Dict[str, np.ndarray], index: int) -> tuple[int, ...]:
    return tuple(int(data[key][index]) for key in ROW_METADATA_KEYS)


def build_row_index_map(data: Dict[str, np.ndarray]) -> dict[tuple[int, ...], int]:
    mapping: dict[tuple[int, ...], int] = {}
    for index in range(int(data["y"].shape[0])):
        key = row_key_from_dataset(data, index)
        if key in mapping:
            raise ValueError(f"Duplicate metadata row detected for key={key}")
        mapping[key] = int(index)
    return mapping


def find_source_indices(full_data: Dict[str, np.ndarray], subset_data: Dict[str, np.ndarray]) -> np.ndarray:
    if "source_index" in subset_data:
        source_indices = subset_data["source_index"].astype(np.int64)
        if source_indices.ndim != 1:
            raise ValueError(f"source_index must be 1D, got {source_indices.shape}")
        for row_index, source_index in enumerate(source_indices.tolist()):
            full_key = row_key_from_dataset(full_data, int(source_index))
            subset_key = row_key_from_dataset(subset_data, int(row_index))
            if full_key != subset_key:
                raise AssertionError(
                    f"source_index mismatch at subset row {row_index}: full={full_key}, subset={subset_key}"
                )
        return source_indices

    row_map = build_row_index_map(full_data)
    source_indices = []
    for row_index in range(int(subset_data["y"].shape[0])):
        key = row_key_from_dataset(subset_data, row_index)
        if key not in row_map:
            raise KeyError(f"Calibration row {key} is not present in the full dataset.")
        source_indices.append(int(row_map[key]))
    return np.asarray(source_indices, dtype=np.int64)


def select_dataset_rows(data: Dict[str, np.ndarray], indices: Sequence[int]) -> Dict[str, np.ndarray]:
    selected_indices = np.asarray(indices, dtype=np.int64)
    selected = {}
    num_rows = int(data["y"].shape[0])
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and int(value.shape[0]) == num_rows:
            selected[key] = value[selected_indices]
        else:
            selected[key] = value
    return selected


def save_subset_dataset(dataset: Dict[str, np.ndarray], output_path: str, summary: Dict[str, object]) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved calibration dataset to {output_path}")
    print(f"[INFO] Saved calibration summary to {summary_path}")


def default_calibration_output_path(
    dataset_path: str,
    subject_id: int,
    gesture_ids: Sequence[int],
    repetition_ids: Sequence[int],
) -> str:
    dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
    gesture_tag = "-".join(str(int(value)) for value in gesture_ids)
    repetition_tag = "-".join(str(int(value)) for value in repetition_ids) if repetition_ids else "all"
    return os.path.join(
        "data",
        "ninapro_db5",
        "calibration",
        f"{dataset_stem}_subject{int(subject_id)}_gestures{gesture_tag}_reps{repetition_tag}.npz",
    )


def split_calibration_train_val(
    data: Dict[str, np.ndarray],
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    if float(val_fraction) == 0.0:
        all_indices = np.arange(int(data["y"].shape[0]), dtype=np.int64)
        return all_indices, np.asarray([], dtype=np.int64)

    train_indices: list[int] = []
    val_indices: list[int] = []
    gesture_ids = data["gesture_id"].astype(np.int64)
    for gesture_id in np.unique(gesture_ids):
        gesture_indices = np.flatnonzero(gesture_ids == int(gesture_id)).astype(np.int64)
        if int(gesture_indices.shape[0]) < 2:
            train_indices.extend(int(index) for index in gesture_indices.tolist())
            continue
        num_val = max(1, int(round(float(gesture_indices.shape[0]) * float(val_fraction))))
        num_val = min(num_val, int(gesture_indices.shape[0]) - 1)
        split_point = int(gesture_indices.shape[0]) - num_val
        train_indices.extend(int(index) for index in gesture_indices[:split_point].tolist())
        val_indices.extend(int(index) for index in gesture_indices[split_point:].tolist())

    return np.asarray(sorted(train_indices), dtype=np.int64), np.asarray(sorted(val_indices), dtype=np.int64)


def parse_mix_ratio(text: str) -> tuple[int, int]:
    stripped = str(text).strip()
    if ":" not in stripped:
        raise ValueError(f"Mix ratio must be in original:calibration form, got {text}")
    left_text, right_text = stripped.split(":", 1)
    original_ratio = int(left_text)
    calibration_ratio = int(right_text)
    if original_ratio < 0 or calibration_ratio <= 0:
        raise ValueError(f"Invalid mix ratio {text}; expected non-negative original and positive calibration parts.")
    return original_ratio, calibration_ratio


def sample_original_indices(
    pool_indices: Sequence[int],
    calibration_size: int,
    ratio_text: str,
    seed: int,
) -> np.ndarray:
    original_ratio, calibration_ratio = parse_mix_ratio(ratio_text)
    pool = np.asarray(pool_indices, dtype=np.int64)
    if original_ratio == 0 or calibration_size <= 0 or int(pool.shape[0]) == 0:
        return np.asarray([], dtype=np.int64)
    requested = int(round(float(calibration_size) * float(original_ratio) / float(calibration_ratio)))
    requested = min(requested, int(pool.shape[0]))
    if requested <= 0:
        return np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(pool, size=requested, replace=False)
    return np.asarray(sorted(int(index) for index in chosen.tolist()), dtype=np.int64)


def build_subject_eval_indices(
    data: Dict[str, np.ndarray],
    subject_id: int,
    gesture_ids: Sequence[int] | None,
    repetition_ids: Sequence[int],
    exclude_indices: Iterable[int] | None = None,
) -> np.ndarray:
    mask = data["subject_id"].astype(np.int64) == int(subject_id)
    if gesture_ids is not None:
        mask &= np.isin(data["gesture_id"].astype(np.int64), [int(value) for value in gesture_ids])
    if repetition_ids:
        mask &= np.isin(data["repetition_id"].astype(np.int64), [int(value) for value in repetition_ids])
    indices = np.flatnonzero(mask).astype(np.int64)
    if exclude_indices is not None:
        exclusion = set(int(value) for value in exclude_indices)
        indices = np.asarray([int(index) for index in indices.tolist() if int(index) not in exclusion], dtype=np.int64)
    return indices


def per_class_accuracy_rows(confusion: np.ndarray, class_gesture_ids: Sequence[int]) -> list[dict[str, object]]:
    supports = confusion.sum(axis=1).astype(np.int64)
    correct = np.diag(confusion).astype(np.int64)
    accuracies = np.divide(
        correct,
        supports,
        out=np.zeros_like(correct, dtype=np.float64),
        where=supports > 0,
    )
    rows = []
    for class_index, gesture_id in enumerate(class_gesture_ids):
        rows.append(
            {
                "class_index": int(class_index),
                "gesture_id": int(gesture_id),
                "support": int(supports[class_index]),
                "correct": int(correct[class_index]),
                "accuracy": float(accuracies[class_index]),
            }
        )
    return rows
