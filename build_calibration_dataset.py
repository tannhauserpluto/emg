from __future__ import annotations

"""Build a small target-user calibration subset from a DB5 raw-window dataset.

Examples
--------
python build_calibration_dataset.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz --subject-id 10 --gestures 2,7,8,9,10

python build_calibration_dataset.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz --subject-id 10 --gestures 2,7,8,9,10 --repetitions 1,3,4 --max-windows-per-gesture 120 --out data/ninapro_db5/calibration/db5_s1-10_subject10_focus_g2-7-8-9-10.npz
"""

import argparse
import json

import numpy as np

from calibration_utils import default_calibration_output_path, save_subset_dataset, select_dataset_rows
from dl_dataset import load_window_dataset, parse_int_list


def _distribute_evenly(total_to_select: int, repetition_order: list[int], repetition_capacities: dict[int, int]) -> dict[int, int]:
    quotas = {int(repetition_id): 0 for repetition_id in repetition_order}
    remaining = int(total_to_select)

    while remaining > 0:
        progressed = False
        for repetition_id in repetition_order:
            repetition_id = int(repetition_id)
            if quotas[repetition_id] >= int(repetition_capacities[repetition_id]):
                continue
            quotas[repetition_id] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    return quotas


def select_calibration_indices(
    data: dict[str, np.ndarray],
    subject_id: int,
    gesture_ids: list[int],
    repetition_ids: list[int],
    max_windows_per_gesture: int,
) -> np.ndarray:
    subject_mask = data["subject_id"].astype(np.int64) == int(subject_id)
    gesture_mask = np.isin(data["gesture_id"].astype(np.int64), [int(value) for value in gesture_ids])
    repetition_mask = np.ones_like(subject_mask, dtype=bool)
    if repetition_ids:
        repetition_mask = np.isin(data["repetition_id"].astype(np.int64), [int(value) for value in repetition_ids])
    candidate_indices = np.flatnonzero(subject_mask & gesture_mask & repetition_mask).astype(np.int64)

    if int(candidate_indices.shape[0]) == 0:
        raise RuntimeError(
            f"No windows found for subject={subject_id}, gestures={gesture_ids}, repetitions={repetition_ids or 'all'}."
        )

    if int(max_windows_per_gesture) <= 0:
        return candidate_indices

    selected_indices: list[int] = []
    selected_index_set: set[int] = set()

    for gesture_id in gesture_ids:
        gesture_id = int(gesture_id)
        gesture_indices = [
            int(index)
            for index in candidate_indices.tolist()
            if int(data["gesture_id"][int(index)]) == gesture_id
        ]
        if not gesture_indices:
            continue

        if repetition_ids:
            repetition_order = [int(repetition_id) for repetition_id in repetition_ids]
        else:
            repetition_order = sorted(
                {
                    int(data["repetition_id"][int(index)])
                    for index in gesture_indices
                }
            )

        repetition_to_indices = {
            int(repetition_id): [
                int(index)
                for index in gesture_indices
                if int(data["repetition_id"][int(index)]) == int(repetition_id)
            ]
            for repetition_id in repetition_order
        }
        repetition_capacities = {
            int(repetition_id): int(len(repetition_to_indices[int(repetition_id)]))
            for repetition_id in repetition_order
        }

        available_total = int(sum(repetition_capacities.values()))
        total_to_select = min(int(max_windows_per_gesture), available_total)
        quotas = _distribute_evenly(total_to_select, repetition_order, repetition_capacities)

        for repetition_id in repetition_order:
            quota = int(quotas[int(repetition_id)])
            if quota <= 0:
                continue
            selected_index_set.update(repetition_to_indices[int(repetition_id)][:quota])

    for index in candidate_indices.tolist():
        if int(index) in selected_index_set:
            selected_indices.append(int(index))

    return np.asarray(selected_indices, dtype=np.int64)


def build_calibration_dataset(
    data: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    source_dataset_path: str,
) -> dict[str, np.ndarray]:
    subset = select_dataset_rows(data, selected_indices)
    subset["source_index"] = selected_indices.astype(np.int64)
    subset["source_dataset_path"] = np.asarray(source_dataset_path)
    return subset


def build_summary(dataset: dict[str, np.ndarray], subject_id: int, gesture_ids: list[int], repetition_ids: list[int]) -> dict[str, object]:
    summary = {
        "num_windows": int(dataset["x"].shape[0]),
        "window_shape": [int(dataset["x"].shape[1]), int(dataset["x"].shape[2])],
        "subject_id": int(subject_id),
        "selected_gestures": [int(value) for value in gesture_ids],
        "selected_repetitions": [int(value) for value in repetition_ids],
        "gestures_present": [int(value) for value in np.unique(dataset["gesture_id"]).astype(np.int64).tolist()],
        "repetitions_present": [int(value) for value in np.unique(dataset["repetition_id"]).astype(np.int64).tolist()],
        "source_indices_range": [
            int(np.min(dataset["source_index"]).item()),
            int(np.max(dataset["source_index"]).item()),
        ],
        "windows_per_gesture": {
            str(int(gesture_id)): int(np.sum(dataset["gesture_id"].astype(np.int64) == int(gesture_id)))
            for gesture_id in np.unique(dataset["gesture_id"]).astype(np.int64)
        },
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small target-user calibration subset from a DB5 raw-window dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a DB5 raw-window dataset .npz")
    parser.add_argument("--subject-id", type=int, required=True, help="Target user subject ID")
    parser.add_argument("--gestures", type=str, required=True, help="Subset of gestures, for example 2,7,8,9,10")
    parser.add_argument("--repetitions", type=str, default="1,3,4", help="Calibration repetitions, default keeps repetition_holdout train reps")
    parser.add_argument("--max-windows-per-gesture", type=int, default=100, help="Limit windows per selected gesture while preserving order")
    parser.add_argument("--out", type=str, default="", help="Output .npz path")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    data = load_window_dataset(args.dataset)
    gesture_ids = parse_int_list(args.gestures)
    repetition_ids = parse_int_list(args.repetitions)
    if not gesture_ids:
        raise ValueError("--gestures must not be empty")

    selected_indices = select_calibration_indices(
        data=data,
        subject_id=int(args.subject_id),
        gesture_ids=gesture_ids,
        repetition_ids=repetition_ids,
        max_windows_per_gesture=int(args.max_windows_per_gesture),
    )
    dataset = build_calibration_dataset(data, selected_indices, args.dataset)

    discovered_subjects = np.unique(dataset["subject_id"]).astype(np.int64).tolist()
    if discovered_subjects != [int(args.subject_id)]:
        raise AssertionError(
            f"Calibration dataset subjects {discovered_subjects} do not match requested subject {int(args.subject_id)}"
        )

    output_path = args.out or default_calibration_output_path(
        dataset_path=args.dataset,
        subject_id=int(args.subject_id),
        gesture_ids=gesture_ids,
        repetition_ids=repetition_ids,
    )
    summary = build_summary(dataset, int(args.subject_id), gesture_ids, repetition_ids)
    save_subset_dataset(dataset, output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
