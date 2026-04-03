from __future__ import annotations

"""Dataset, split, and normalization utilities for DB5 deep learning experiments."""

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_WINDOW_KEYS = (
    "x",
    "y",
    "subject_id",
    "exercise_id",
    "gesture_id",
    "repetition_id",
    "window_start",
    "window_end",
)

REQUIRED_FEATURE_KEYS = (
    "features",
    "y",
    "subject_id",
    "exercise_id",
    "gesture_id",
    "repetition_id",
    "window_start",
    "window_end",
)


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


def infer_class_gesture_ids(data: Dict[str, np.ndarray], require_contiguous: bool = True) -> np.ndarray:
    class_ids = data["y"].astype(np.int64)
    gesture_ids = data["gesture_id"].astype(np.int64)

    mapping: Dict[int, int] = {}
    for class_id, gesture_id in zip(class_ids.tolist(), gesture_ids.tolist()):
        existing = mapping.get(int(class_id))
        if existing is None:
            mapping[int(class_id)] = int(gesture_id)
        elif existing != int(gesture_id):
            raise ValueError(f"Class {class_id} maps to multiple gestures: {existing} and {gesture_id}")

    expected_class_ids = list(range(int(class_ids.max()) + 1))
    if require_contiguous:
        if sorted(mapping.keys()) != expected_class_ids:
            raise ValueError(
                f"Class IDs must be contiguous from 0. Found {sorted(mapping.keys())}, expected {expected_class_ids}."
            )
        return np.asarray([mapping[class_id] for class_id in expected_class_ids], dtype=np.int64)

    if "class_gesture_ids" in data:
        stored = data["class_gesture_ids"].astype(np.int64)
        for class_id, gesture_id in mapping.items():
            if int(class_id) >= int(stored.shape[0]):
                raise ValueError(
                    f"Sparse class ID {class_id} exceeds stored class_gesture_ids length {stored.shape[0]}."
                )
            if int(stored[int(class_id)]) != int(gesture_id):
                raise ValueError(
                    f"Stored class_gesture_ids[{class_id}]={int(stored[int(class_id)])} does not match inferred gesture {gesture_id}."
                )
        return stored

    raise ValueError("Sparse class IDs require stored class_gesture_ids in the dataset.")


def load_window_dataset(dataset_path: str, allow_sparse_labels: bool = False) -> Dict[str, np.ndarray]:
    loaded = np.load(dataset_path, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}

    for key in REQUIRED_WINDOW_KEYS:
        if key not in data:
            raise KeyError(f"Dataset {dataset_path} is missing required key: {key}")

    if data["x"].ndim != 3:
        raise ValueError(f"Expected x to have shape [N, T, C], got {data['x'].shape}")

    num_samples = int(data["x"].shape[0])
    for key in REQUIRED_WINDOW_KEYS[1:]:
        if int(data[key].shape[0]) != num_samples:
            raise ValueError(
                f"Key {key} has {data[key].shape[0]} rows but x has {num_samples} windows."
            )

    if "class_gesture_ids" not in data:
        data["class_gesture_ids"] = infer_class_gesture_ids(data, require_contiguous=not allow_sparse_labels)
    else:
        inferred = infer_class_gesture_ids(data, require_contiguous=not allow_sparse_labels)
        stored = data["class_gesture_ids"].astype(np.int64)
        if stored.shape != inferred.shape or not np.array_equal(stored, inferred):
            raise ValueError(
                f"Stored class_gesture_ids {stored.tolist()} do not match inferred mapping {inferred.tolist()}."
            )

    data["x"] = data["x"].astype(np.float32, copy=False)
    data["y"] = data["y"].astype(np.int64, copy=False)
    return data


def load_feature_dataset(feature_path: str) -> Dict[str, np.ndarray]:
    loaded = np.load(feature_path, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}

    for key in REQUIRED_FEATURE_KEYS:
        if key not in data:
            raise KeyError(f"Feature dataset {feature_path} is missing required key: {key}")

    if data["features"].ndim != 2:
        raise ValueError(f"Expected features to have shape [N, D], got {data['features'].shape}")

    num_samples = int(data["features"].shape[0])
    for key in REQUIRED_FEATURE_KEYS[1:]:
        if int(data[key].shape[0]) != num_samples:
            raise ValueError(
                f"Key {key} has {data[key].shape[0]} rows but features has {num_samples} rows."
            )

    if "class_gesture_ids" not in data:
        data["class_gesture_ids"] = infer_class_gesture_ids(data)
    else:
        inferred = infer_class_gesture_ids(data)
        stored = data["class_gesture_ids"].astype(np.int64)
        if stored.shape != inferred.shape or not np.array_equal(stored, inferred):
            raise ValueError(
                f"Stored class_gesture_ids {stored.tolist()} do not match inferred mapping {inferred.tolist()}."
            )

    data["features"] = data["features"].astype(np.float32, copy=False)
    data["y"] = data["y"].astype(np.int64, copy=False)
    return data


def assert_feature_alignment(window_data: Dict[str, np.ndarray], feature_data: Dict[str, np.ndarray]) -> None:
    metadata_keys = (
        "y",
        "subject_id",
        "exercise_id",
        "gesture_id",
        "repetition_id",
        "window_start",
        "window_end",
    )

    if int(window_data["x"].shape[0]) != int(feature_data["features"].shape[0]):
        raise AssertionError(
            f"Window dataset has {window_data['x'].shape[0]} rows but feature dataset has {feature_data['features'].shape[0]} rows."
        )

    for key in metadata_keys:
        if key not in feature_data:
            raise KeyError(f"Feature dataset is missing alignment key: {key}")
        if not np.array_equal(window_data[key], feature_data[key]):
            raise AssertionError(f"Feature dataset key {key} is not aligned with the raw window dataset.")

    if not np.array_equal(window_data["class_gesture_ids"], feature_data["class_gesture_ids"]):
        raise AssertionError("Feature dataset class_gesture_ids do not match the raw window dataset.")


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class ChannelwiseNormalizer:
    mean: np.ndarray
    std: np.ndarray
    per_channel: bool = True

    @classmethod
    def fit(cls, windows: np.ndarray, per_channel: bool = True) -> "ChannelwiseNormalizer":
        if windows.ndim != 3:
            raise ValueError(f"Expected windows to have shape [N, T, C], got {windows.shape}")
        if per_channel:
            mean = windows.mean(axis=(0, 1)).astype(np.float32)
            std = windows.std(axis=(0, 1)).astype(np.float32)
        else:
            mean = np.asarray([windows.mean()], dtype=np.float32)
            std = np.asarray([windows.std()], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std, per_channel=bool(per_channel))

    def transform_window(self, window: np.ndarray) -> np.ndarray:
        return ((window.astype(np.float32) - self.mean[None, :]) / self.std[None, :]).astype(np.float32)

    def transform_windows(self, windows: np.ndarray) -> np.ndarray:
        return ((windows.astype(np.float32) - self.mean[None, None, :]) / self.std[None, None, :]).astype(np.float32)

    def to_state(self) -> Dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "per_channel": bool(self.per_channel)}

    @classmethod
    def from_state(cls, state: Dict[str, Sequence[float]]) -> "ChannelwiseNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            per_channel=bool(state.get("per_channel", True)),
        )


@dataclass
class VectorNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feature_matrix: np.ndarray) -> "VectorNormalizer":
        if feature_matrix.ndim != 2:
            raise ValueError(f"Expected feature_matrix to have shape [N, D], got {feature_matrix.shape}")
        mean = feature_matrix.mean(axis=0).astype(np.float32)
        std = feature_matrix.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform_vector(self, feature_vector: np.ndarray) -> np.ndarray:
        return ((feature_vector.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def transform_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        return ((feature_matrix.astype(np.float32) - self.mean[None, :]) / self.std[None, :]).astype(np.float32)

    def to_state(self) -> Dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: Dict[str, Sequence[float]]) -> "VectorNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


@dataclass
class EMGAugmenter:
    scale: float = 0.0
    noise: float = 0.0
    shift: int = 0
    channel_dropout: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.scale > 0 or self.noise > 0 or self.shift > 0 or self.channel_dropout > 0)

    def apply(self, window: np.ndarray) -> np.ndarray:
        augmented = window.astype(np.float32, copy=True)

        if self.scale > 0:
            scale_factors = np.random.uniform(
                low=1.0 - float(self.scale),
                high=1.0 + float(self.scale),
                size=(1, augmented.shape[1]),
            ).astype(np.float32)
            augmented = augmented * scale_factors

        if self.noise > 0:
            noise = np.random.normal(loc=0.0, scale=float(self.noise), size=augmented.shape).astype(np.float32)
            augmented = augmented + noise

        if self.shift > 0:
            max_shift = min(int(self.shift), max(int(augmented.shape[0]) - 1, 0))
            shift_value = int(np.random.randint(-max_shift, max_shift + 1)) if max_shift > 0 else 0
            if shift_value != 0:
                shifted = np.zeros_like(augmented)
                if shift_value > 0:
                    shifted[shift_value:] = augmented[:-shift_value]
                else:
                    shifted[:shift_value] = augmented[-shift_value:]
                augmented = shifted

        if self.channel_dropout > 0:
            keep_mask = (np.random.rand(augmented.shape[1]) >= float(self.channel_dropout)).astype(np.float32)
            if not np.any(keep_mask):
                keep_mask[int(np.random.randint(0, augmented.shape[1]))] = 1.0
            augmented = augmented * keep_mask[None, :]

        return augmented.astype(np.float32)


def reconstruct_window_to_image(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Expected window to have shape [T, C], got {window.shape}")
    return np.expand_dims(window.astype(np.float32, copy=False), axis=0).astype(np.float32, copy=False)


class WindowedEMGDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
        normalizer: ChannelwiseNormalizer | None = None,
        feature_matrix: np.ndarray | None = None,
        feature_normalizer: VectorNormalizer | None = None,
        augmenter: EMGAugmenter | None = None,
        image_input: bool = False,
    ) -> None:
        self.windows = windows
        self.labels = labels.astype(np.int64, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalizer = normalizer
        self.feature_matrix = feature_matrix
        self.feature_normalizer = feature_normalizer
        self.augmenter = augmenter
        self.image_input = bool(image_input)

        if self.feature_matrix is not None and int(self.feature_matrix.shape[0]) != int(self.windows.shape[0]):
            raise ValueError(
                f"Feature matrix rows {self.feature_matrix.shape[0]} do not match window rows {self.windows.shape[0]}."
            )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        sample_index = int(self.indices[index])
        window = self.windows[sample_index]
        if self.augmenter is not None and self.augmenter.enabled:
            window = self.augmenter.apply(window)
        if self.normalizer is not None:
            window = self.normalizer.transform_window(window)
        if self.image_input:
            window = reconstruct_window_to_image(window)
        label = int(self.labels[sample_index])
        window_tensor = torch.from_numpy(window.copy())
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.feature_matrix is None:
            return window_tensor, label_tensor

        feature_vector = self.feature_matrix[sample_index]
        if self.feature_normalizer is not None:
            feature_vector = self.feature_normalizer.transform_vector(feature_vector)
        feature_tensor = torch.from_numpy(feature_vector.copy())
        return window_tensor, feature_tensor, label_tensor


def _ensure_disjoint(name_to_values: Dict[str, Sequence[int]]) -> None:
    names = list(name_to_values.keys())
    for left_index, left_name in enumerate(names):
        left_values = set(int(value) for value in name_to_values[left_name])
        for right_name in names[left_index + 1 :]:
            right_values = set(int(value) for value in name_to_values[right_name])
            overlap = sorted(left_values & right_values)
            if overlap:
                raise ValueError(f"Overlap between {left_name} and {right_name}: {overlap}")


def build_split_indices(
    data: Dict[str, np.ndarray],
    split_mode: str,
    train_reps: Sequence[int] | None = None,
    val_reps: Sequence[int] | None = None,
    test_reps: Sequence[int] | None = None,
    train_subjects: Sequence[int] | None = None,
    val_subjects: Sequence[int] | None = None,
    test_subjects: Sequence[int] | None = None,
) -> tuple[SplitIndices, Dict[str, object]]:
    split_mode = str(split_mode)

    if split_mode == "repetition_holdout":
        train_reps = parse_int_list(train_reps or [])
        val_reps = parse_int_list(val_reps or [])
        test_reps = parse_int_list(test_reps or [])
        _ensure_disjoint({"train_reps": train_reps, "val_reps": val_reps, "test_reps": test_reps})

        available_reps = [int(value) for value in np.unique(data["repetition_id"]).tolist()]
        assigned_reps = set(train_reps) | set(val_reps) | set(test_reps)
        missing_reps = sorted(set(available_reps) - assigned_reps)
        if missing_reps:
            raise ValueError(f"Repetitions {missing_reps} are present in the dataset but not assigned to any split.")

        train_indices = np.flatnonzero(np.isin(data["repetition_id"], train_reps))
        val_indices = np.flatnonzero(np.isin(data["repetition_id"], val_reps))
        test_indices = np.flatnonzero(np.isin(data["repetition_id"], test_reps))

        split_config = {
            "split_mode": split_mode,
            "train_reps": [int(value) for value in train_reps],
            "val_reps": [int(value) for value in val_reps],
            "test_reps": [int(value) for value in test_reps],
        }
    elif split_mode == "subject_holdout":
        train_subjects = parse_int_list(train_subjects or [])
        val_subjects = parse_int_list(val_subjects or [])
        test_subjects = parse_int_list(test_subjects or [])
        _ensure_disjoint(
            {
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
            }
        )

        available_subjects = [int(value) for value in np.unique(data["subject_id"]).tolist()]
        assigned_subjects = set(train_subjects) | set(val_subjects) | set(test_subjects)
        missing_subjects = sorted(set(available_subjects) - assigned_subjects)
        if missing_subjects:
            raise ValueError(
                f"Subjects {missing_subjects} are present in the dataset but not assigned to any split."
            )

        train_indices = np.flatnonzero(np.isin(data["subject_id"], train_subjects))
        val_indices = np.flatnonzero(np.isin(data["subject_id"], val_subjects))
        test_indices = np.flatnonzero(np.isin(data["subject_id"], test_subjects))

        split_config = {
            "split_mode": split_mode,
            "train_subjects": [int(value) for value in train_subjects],
            "val_subjects": [int(value) for value in val_subjects],
            "test_subjects": [int(value) for value in test_subjects],
        }
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    split_indices = SplitIndices(
        train=train_indices.astype(np.int64),
        val=val_indices.astype(np.int64),
        test=test_indices.astype(np.int64),
    )

    for split_name, indices in {
        "train": split_indices.train,
        "val": split_indices.val,
        "test": split_indices.test,
    }.items():
        if int(indices.shape[0]) == 0:
            raise ValueError(f"Split {split_name} is empty.")

    return split_indices, split_config


def assert_no_overlap(data: Dict[str, np.ndarray], split_indices: SplitIndices) -> None:
    index_sets = {
        "train": set(int(value) for value in split_indices.train.tolist()),
        "val": set(int(value) for value in split_indices.val.tolist()),
        "test": set(int(value) for value in split_indices.test.tolist()),
    }

    names = list(index_sets.keys())
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = index_sets[left_name] & index_sets[right_name]
            if overlap:
                raise AssertionError(f"Window overlap detected between {left_name} and {right_name}: {sorted(overlap)[:5]}")

    group_sets = {
        split_name: {
            (int(subject_id), int(repetition_id))
            for subject_id, repetition_id in zip(
                data["subject_id"][indices].tolist(),
                data["repetition_id"][indices].tolist(),
            )
        }
        for split_name, indices in {
            "train": split_indices.train,
            "val": split_indices.val,
            "test": split_indices.test,
        }.items()
    }

    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = group_sets[left_name] & group_sets[right_name]
            if overlap:
                raise AssertionError(
                    f"Leakage detected between {left_name} and {right_name} on subject_id/repetition_id groups: {sorted(overlap)[:5]}"
                )


def _split_detail(data: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, object]:
    subject_ids = data["subject_id"][indices].astype(np.int64)
    exercise_ids = data["exercise_id"][indices].astype(np.int64)
    gesture_ids = data["gesture_id"][indices].astype(np.int64)
    repetition_ids = data["repetition_id"][indices].astype(np.int64)

    group_pairs = sorted(
        {
            (int(subject_id), int(repetition_id))
            for subject_id, repetition_id in zip(subject_ids.tolist(), repetition_ids.tolist())
        }
    )

    windows_per_subject = {
        str(int(subject_id)): int(np.sum(subject_ids == subject_id))
        for subject_id in np.unique(subject_ids)
    }
    windows_per_gesture = {
        str(int(gesture_id)): int(np.sum(gesture_ids == gesture_id))
        for gesture_id in np.unique(gesture_ids)
    }

    return {
        "num_windows": int(indices.shape[0]),
        "num_groups": int(len(group_pairs)),
        "subjects": [int(value) for value in np.unique(subject_ids).tolist()],
        "exercises": [int(value) for value in np.unique(exercise_ids).tolist()],
        "gestures": [int(value) for value in np.unique(gesture_ids).tolist()],
        "repetitions": [int(value) for value in np.unique(repetition_ids).tolist()],
        "group_pairs": [
            {"subject_id": int(subject_id), "repetition_id": int(repetition_id)}
            for subject_id, repetition_id in group_pairs
        ],
        "windows_per_subject": windows_per_subject,
        "windows_per_gesture": windows_per_gesture,
    }


def make_split_summary(
    data: Dict[str, np.ndarray],
    split_indices: SplitIndices,
    split_config: Dict[str, object],
) -> Dict[str, object]:
    dataset_summary = {
        "num_windows": int(data["x"].shape[0]),
        "window_shape": [int(data["x"].shape[1]), int(data["x"].shape[2])],
        "num_classes": int(len(data["class_gesture_ids"])),
        "class_gesture_ids": [int(value) for value in data["class_gesture_ids"].tolist()],
        "subjects": [int(value) for value in np.unique(data["subject_id"]).tolist()],
        "exercises": [int(value) for value in np.unique(data["exercise_id"]).tolist()],
        "repetitions": [int(value) for value in np.unique(data["repetition_id"]).tolist()],
        "sampling_rate_hz": int(np.asarray(data.get("sampling_rate_hz", np.asarray(0))).item()),
        "window_size_samples": int(np.asarray(data.get("window_size_samples", np.asarray(data["x"].shape[1]))).item()),
        "step_size_samples": int(np.asarray(data.get("step_size_samples", np.asarray(0))).item()),
    }

    summary = {
        "dataset": dataset_summary,
        "split_config": split_config,
        "splits": {
            "train": _split_detail(data, split_indices.train),
            "val": _split_detail(data, split_indices.val),
            "test": _split_detail(data, split_indices.test),
        },
    }
    return summary


def print_split_summary(summary: Dict[str, object]) -> None:
    print("[INFO] Split summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
