from __future__ import annotations

"""Fine-tune a pretrained DB5 model with target-user calibration data.

Examples
--------
python build_calibration_dataset.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz --subject-id 10 --gestures 2,7,8,9,10 --repetitions 1,3,4 --max-windows-per-gesture 120

python calibrate_dl.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz --calibration-dataset data/ninapro_db5/calibration/db5_s1-10_g1-10_win400_step50_subject10_gestures2-7-8-9-10_reps1-3-4.npz --pretrained-checkpoint artifacts/dl/db5_s1-10_win400_dualres_xception2d/best.pt --finetune-mode upper_layers --mix-ratio 3:1 --run-dir artifacts/dl/calibration_subject10_dualres
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from calibration_utils import (
    build_subject_eval_indices,
    find_source_indices,
    parse_mix_ratio,
    per_class_accuracy_rows,
    sample_original_indices,
    split_calibration_train_val,
)
from dl_dataset import (
    EMGAugmenter,
    WindowedEMGDataset,
    assert_no_overlap,
    build_split_indices,
    load_window_dataset,
    make_split_summary,
    parse_int_list,
    print_split_summary,
)
from train_dl import (
    autocast_context,
    build_model_from_config,
    evaluate_model,
    forward_model,
    move_batch_to_device,
    print_runtime_info,
    requires_pseudo_image_input,
    resolve_device,
    safe_torch_load,
    save_confusion_matrix_outputs,
    save_history_csv,
    save_json,
    save_per_class_accuracy_outputs,
    set_seed,
)


def infer_single_subject(calibration_data: Dict[str, np.ndarray]) -> int:
    subjects = np.unique(calibration_data["subject_id"]).astype(np.int64).tolist()
    if len(subjects) != 1:
        raise ValueError(f"Calibration dataset must contain exactly one subject, found {subjects}")
    return int(subjects[0])


def infer_target_gestures(calibration_data: Dict[str, np.ndarray], gesture_text: str) -> list[int]:
    if str(gesture_text).strip():
        gesture_ids = parse_int_list(gesture_text)
        if not gesture_ids:
            raise ValueError("--target-gestures was provided but parsed to an empty set.")
        return gesture_ids
    return [int(value) for value in np.unique(calibration_data["gesture_id"]).astype(np.int64).tolist()]


def resolve_effective_finetune_mode(finetune_mode: str, freeze_backbone: bool) -> str:
    if freeze_backbone and finetune_mode != "head_only":
        raise ValueError("--freeze-backbone is only compatible with --finetune-mode head_only.")
    if freeze_backbone:
        return "head_only"
    return str(finetune_mode)


def trainable_prefixes_for_mode(model_name: str, finetune_mode: str) -> tuple[str, ...]:
    if finetune_mode == "head_only":
        return ("classifier",)
    if finetune_mode != "upper_layers":
        raise ValueError(f"Unsupported fine-tune mode: {finetune_mode}")

    mapping = {
        "dualres_xception2d": ("block2", "block3", "exit", "skip1", "skip2", "classifier"),
        "xception2d": ("block2", "block3", "exit", "classifier"),
        "cnn2d_baseline": ("features.4", "features.5", "features.6", "features.7", "features.8", "features.9", "classifier"),
        "cnn_lstm": ("frontend.4", "frontend.5", "frontend.6", "frontend.7", "temporal_model", "classifier"),
    }
    if model_name not in mapping:
        raise ValueError(f"Fine-tune mode upper_layers is not configured for model {model_name}")
    return mapping[model_name]


def apply_finetune_mode(model: nn.Module, model_name: str, finetune_mode: str) -> list[str]:
    prefixes = trainable_prefixes_for_mode(model_name, finetune_mode)
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError(f"No trainable parameters selected for mode={finetune_mode} model={model_name}")
    return trainable_names


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            has_trainable_parameters = any(parameter.requires_grad for parameter in module.parameters(recurse=False))
            if not has_trainable_parameters:
                module.eval()


def build_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )


def train_calibration_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    model.train()
    set_frozen_batchnorm_eval(model)
    running_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        window_batch, feature_batch, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            logits = forward_model(model, window_batch, feature_batch)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(labels.shape[0])
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def save_gesture_improvements(
    before_confusion: np.ndarray,
    after_confusion: np.ndarray,
    class_gesture_ids: Sequence[int],
    target_gestures: Sequence[int],
    output_dir: str,
) -> None:
    before_rows = {int(row["gesture_id"]): row for row in per_class_accuracy_rows(before_confusion, class_gesture_ids)}
    after_rows = {int(row["gesture_id"]): row for row in per_class_accuracy_rows(after_confusion, class_gesture_ids)}
    rows = []
    for gesture_id in target_gestures:
        before_row = before_rows[int(gesture_id)]
        after_row = after_rows[int(gesture_id)]
        rows.append(
            {
                "gesture_id": int(gesture_id),
                "before_accuracy": float(before_row["accuracy"]),
                "after_accuracy": float(after_row["accuracy"]),
                "delta_accuracy": float(after_row["accuracy"] - before_row["accuracy"]),
                "before_support": int(before_row["support"]),
                "after_support": int(after_row["support"]),
            }
        )

    rows.sort(key=lambda row: row["gesture_id"])
    csv_path = os.path.join(output_dir, "target_gesture_improvements.csv")
    json_path = os.path.join(output_dir, "target_gesture_improvements.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()) if rows else ["gesture_id"])
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, ensure_ascii=False, indent=2)
    print("[INFO] Target gesture improvements:")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained DB5 model with target-user calibration data.")
    parser.add_argument("--dataset", type=str, required=True, help="Full DB5 raw-window dataset")
    parser.add_argument("--calibration-dataset", type=str, required=True, help="Target-user calibration subset built from raw windows")
    parser.add_argument("--pretrained-checkpoint", type=str, required=True, help="Pretrained checkpoint to calibrate")
    parser.add_argument("--target-gestures", type=str, default="", help="Optional override for focus gestures")
    parser.add_argument("--train-reps", type=str, default="1,3,4")
    parser.add_argument("--val-reps", type=str, default="6")
    parser.add_argument("--test-reps", type=str, default="2,5")
    parser.add_argument("--eval-reps", type=str, default="2,5", help="Target-user evaluation repetitions after excluding calibration windows")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--finetune-mode", type=str, choices=["head_only", "upper_layers"], default="head_only")
    parser.add_argument("--mix-ratio", type=str, default="3:1", help="original:calibration mix ratio")
    parser.add_argument("--calibration-val-fraction", type=float, default=0.2)
    parser.add_argument("--calib-augment-scale", type=float, default=0.02)
    parser.add_argument("--calib-augment-noise", type=float, default=0.005)
    parser.add_argument("--calib-augment-shift", type=int, default=1)
    parser.add_argument("--calib-augment-channel-dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-dir", type=str, default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print_runtime_info(device)
    amp_enabled = device.type == "cuda" if args.amp is None else bool(args.amp)
    if amp_enabled and device.type != "cuda":
        raise RuntimeError("AMP was requested, but the selected device is not CUDA. Use --device cuda/auto or pass --no-amp.")
    print(f"[INFO] AMP enabled: {amp_enabled}")

    checkpoint = safe_torch_load(args.pretrained_checkpoint, map_location=device)
    model_name = str(checkpoint["model_name"])
    if model_name == "fusion_tcn_mlp":
        raise ValueError("Calibration workflow currently supports raw-window models only, not fusion_tcn_mlp.")

    full_dataset = load_window_dataset(args.dataset)
    calibration_dataset = load_window_dataset(args.calibration_dataset, allow_sparse_labels=True)
    calibration_source_indices = find_source_indices(full_dataset, calibration_dataset)

    split_indices, split_config = build_split_indices(
        full_dataset,
        split_mode="repetition_holdout",
        train_reps=parse_int_list(args.train_reps),
        val_reps=parse_int_list(args.val_reps),
        test_reps=parse_int_list(args.test_reps),
    )
    assert_no_overlap(full_dataset, split_indices)
    split_summary = make_split_summary(full_dataset, split_indices, split_config)
    print_split_summary(split_summary)

    target_subject = infer_single_subject(calibration_dataset)
    target_gestures = infer_target_gestures(calibration_dataset, args.target_gestures)
    effective_mode = resolve_effective_finetune_mode(args.finetune_mode, bool(args.freeze_backbone))
    parse_mix_ratio(args.mix_ratio)

    if args.run_dir:
        run_dir = args.run_dir
    else:
        checkpoint_stem = os.path.splitext(os.path.basename(args.pretrained_checkpoint))[0]
        run_dir = os.path.join("artifacts", "dl", f"calibration_subject{target_subject}_{checkpoint_stem}_{effective_mode}")
    os.makedirs(run_dir, exist_ok=True)
    save_json(vars(args), os.path.join(run_dir, "calibration_args.json"))
    save_json(split_summary, os.path.join(run_dir, "source_split_summary.json"))

    image_input = requires_pseudo_image_input(model_name)
    class_gesture_ids = [int(value) for value in checkpoint["class_gesture_ids"]]
    num_classes = int(len(class_gesture_ids))
    normalizer = None
    if checkpoint.get("normalizer") is not None:
        from dl_dataset import ChannelwiseNormalizer

        normalizer = ChannelwiseNormalizer.from_state(dict(checkpoint["normalizer"]))
    else:
        raise ValueError("Pretrained checkpoint is missing normalizer state.")

    model = build_model_from_config(model_name, dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    trainable_names = apply_finetune_mode(model, model_name, effective_mode)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"[INFO] Calibration target subject: {target_subject}")
    print(f"[INFO] Focus gestures: {target_gestures}")
    print(f"[INFO] Fine-tune mode: {effective_mode}")
    print(f"[INFO] Trainable parameter tensors: {len(trainable_names)}")
    print(json.dumps(trainable_names, ensure_ascii=False, indent=2))

    calibration_local_train_indices, calibration_local_val_indices = split_calibration_train_val(
        calibration_dataset,
        val_fraction=float(args.calibration_val_fraction),
    )
    calibration_source_train_indices = calibration_source_indices[calibration_local_train_indices]
    calibration_source_val_indices = calibration_source_indices[calibration_local_val_indices]

    original_pool_mask = np.ones(int(split_indices.train.shape[0]), dtype=bool)
    calibration_train_source_set = set(int(value) for value in calibration_source_train_indices.tolist())
    original_pool_indices = np.asarray(
        [
            int(index)
            for index in split_indices.train.tolist()
            if int(index) not in calibration_train_source_set
        ],
        dtype=np.int64,
    )
    sampled_original_indices = sample_original_indices(
        pool_indices=original_pool_indices,
        calibration_size=int(calibration_local_train_indices.shape[0]),
        ratio_text=args.mix_ratio,
        seed=args.seed,
    )

    focus_eval_indices = build_subject_eval_indices(
        full_dataset,
        subject_id=target_subject,
        gesture_ids=target_gestures,
        repetition_ids=parse_int_list(args.eval_reps),
        exclude_indices=calibration_source_indices.tolist(),
    )
    all_eval_indices = build_subject_eval_indices(
        full_dataset,
        subject_id=target_subject,
        gesture_ids=None,
        repetition_ids=parse_int_list(args.eval_reps),
        exclude_indices=calibration_source_indices.tolist(),
    )
    if int(focus_eval_indices.shape[0]) == 0:
        raise RuntimeError("Focus evaluation set is empty after excluding calibration windows.")
    if int(all_eval_indices.shape[0]) == 0:
        raise RuntimeError("All-gesture evaluation set is empty after excluding calibration windows.")

    calibration_augmenter = EMGAugmenter(
        scale=float(args.calib_augment_scale),
        noise=float(args.calib_augment_noise),
        shift=int(args.calib_augment_shift),
        channel_dropout=float(args.calib_augment_channel_dropout),
    )
    print(
        "[INFO] Calibration augmentations only: "
        f"scale={args.calib_augment_scale}, noise={args.calib_augment_noise}, "
        f"shift={args.calib_augment_shift}, channel_dropout={args.calib_augment_channel_dropout}"
    )
    print(
        "[INFO] Mixed-data fine-tuning: "
        f"original_subset={int(sampled_original_indices.shape[0])}, calibration_train={int(calibration_local_train_indices.shape[0])}, "
        f"calibration_val={int(calibration_local_val_indices.shape[0])}, mix_ratio={args.mix_ratio}"
    )

    original_subset_dataset = WindowedEMGDataset(
        full_dataset["x"],
        full_dataset["y"],
        sampled_original_indices,
        normalizer=normalizer,
        image_input=image_input,
    )
    calibration_train_dataset = WindowedEMGDataset(
        calibration_dataset["x"],
        calibration_dataset["y"],
        calibration_local_train_indices,
        normalizer=normalizer,
        augmenter=calibration_augmenter,
        image_input=image_input,
    )
    if int(sampled_original_indices.shape[0]) > 0:
        finetune_train_dataset: Dataset = ConcatDataset([original_subset_dataset, calibration_train_dataset])
    else:
        finetune_train_dataset = calibration_train_dataset

    calibration_val_dataset = WindowedEMGDataset(
        calibration_dataset["x"],
        calibration_dataset["y"],
        calibration_local_val_indices,
        normalizer=normalizer,
        image_input=image_input,
    )
    focus_eval_dataset = WindowedEMGDataset(
        full_dataset["x"],
        full_dataset["y"],
        focus_eval_indices,
        normalizer=normalizer,
        image_input=image_input,
    )
    all_eval_dataset = WindowedEMGDataset(
        full_dataset["x"],
        full_dataset["y"],
        all_eval_indices,
        normalizer=normalizer,
        image_input=image_input,
    )

    train_loader = build_data_loader(finetune_train_dataset, args.batch_size, True, args.num_workers, device)
    monitor_dataset = calibration_val_dataset if int(calibration_local_val_indices.shape[0]) > 0 else calibration_train_dataset
    monitor_loader = build_data_loader(monitor_dataset, args.batch_size, False, args.num_workers, device)
    focus_eval_loader = build_data_loader(focus_eval_dataset, args.batch_size, False, args.num_workers, device)
    all_eval_loader = build_data_loader(all_eval_dataset, args.batch_size, False, args.num_workers, device)

    evaluation_criterion = nn.CrossEntropyLoss()
    before_focus_metrics = evaluate_model(model, focus_eval_loader, evaluation_criterion, device, num_classes, amp_enabled=amp_enabled)
    before_all_metrics = evaluate_model(model, all_eval_loader, evaluation_criterion, device, num_classes, amp_enabled=amp_enabled)

    save_confusion_matrix_outputs(before_focus_metrics["confusion_matrix"], class_gesture_ids, run_dir, "before_focus")
    save_per_class_accuracy_outputs(before_focus_metrics["confusion_matrix"], class_gesture_ids, run_dir, "before_focus")
    save_confusion_matrix_outputs(before_all_metrics["confusion_matrix"], class_gesture_ids, run_dir, "before_all")
    save_per_class_accuracy_outputs(before_all_metrics["confusion_matrix"], class_gesture_ids, run_dir, "before_all")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(args.finetune_lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled)

    history_rows: list[Dict[str, object]] = []
    history_path = os.path.join(run_dir, "history.csv")
    checkpoint_path = os.path.join(run_dir, "best_calibration.pt")
    best_monitor_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        train_loss = train_calibration_epoch(model, train_loader, criterion, optimizer, device, scaler, amp_enabled)
        monitor_metrics = evaluate_model(
            model,
            monitor_loader,
            evaluation_criterion,
            device,
            num_classes,
            amp_enabled=amp_enabled,
        )
        elapsed_seconds = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "monitor_loss": float(monitor_metrics["loss"]),
            "monitor_accuracy": float(monitor_metrics["accuracy"]),
            "monitor_macro_f1": float(monitor_metrics["macro_f1"]),
            "elapsed_seconds": float(elapsed_seconds),
        }
        history_rows.append(row)
        save_history_csv(history_rows, history_path)

        print(
            f"[EPOCH {epoch:03d}] train_loss={train_loss:.6f} "
            f"monitor_loss={monitor_metrics['loss']:.6f} "
            f"monitor_acc={monitor_metrics['accuracy']:.4f} "
            f"monitor_macro_f1={monitor_metrics['macro_f1']:.4f}"
        )

        if float(monitor_metrics["macro_f1"]) > best_monitor_macro_f1 + 1e-8:
            best_monitor_macro_f1 = float(monitor_metrics["macro_f1"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_config": checkpoint["model_config"],
                    "model_state_dict": model.state_dict(),
                    "normalizer": checkpoint["normalizer"],
                    "class_gesture_ids": class_gesture_ids,
                    "dataset_path": args.dataset,
                    "calibration_dataset_path": args.calibration_dataset,
                    "pretrained_checkpoint": args.pretrained_checkpoint,
                    "target_subject": int(target_subject),
                    "target_gestures": [int(value) for value in target_gestures],
                    "finetune_mode": effective_mode,
                    "mix_ratio": args.mix_ratio,
                    "best_epoch": best_epoch,
                    "best_monitor_macro_f1": best_monitor_macro_f1,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"[INFO] Saved new best calibration checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(args.patience):
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    best_checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    best_model = build_model_from_config(str(best_checkpoint["model_name"]), dict(best_checkpoint["model_config"])).to(device)
    best_model.load_state_dict(best_checkpoint["model_state_dict"])

    after_focus_metrics = evaluate_model(best_model, focus_eval_loader, evaluation_criterion, device, num_classes, amp_enabled=amp_enabled)
    after_all_metrics = evaluate_model(best_model, all_eval_loader, evaluation_criterion, device, num_classes, amp_enabled=amp_enabled)

    save_confusion_matrix_outputs(after_focus_metrics["confusion_matrix"], class_gesture_ids, run_dir, "after_focus")
    save_per_class_accuracy_outputs(after_focus_metrics["confusion_matrix"], class_gesture_ids, run_dir, "after_focus")
    save_confusion_matrix_outputs(after_all_metrics["confusion_matrix"], class_gesture_ids, run_dir, "after_all")
    save_per_class_accuracy_outputs(after_all_metrics["confusion_matrix"], class_gesture_ids, run_dir, "after_all")
    save_gesture_improvements(
        before_confusion=before_focus_metrics["confusion_matrix"],
        after_confusion=after_focus_metrics["confusion_matrix"],
        class_gesture_ids=class_gesture_ids,
        target_gestures=target_gestures,
        output_dir=run_dir,
    )

    metrics_summary = {
        "target_subject": int(target_subject),
        "target_gestures": [int(value) for value in target_gestures],
        "finetune_mode": effective_mode,
        "mix_ratio": args.mix_ratio,
        "sample_counts": {
            "original_subset": int(sampled_original_indices.shape[0]),
            "calibration_train": int(calibration_local_train_indices.shape[0]),
            "calibration_val": int(calibration_local_val_indices.shape[0]),
            "focus_eval": int(focus_eval_indices.shape[0]),
            "all_eval": int(all_eval_indices.shape[0]),
        },
        "before": {
            "focus": {
                "loss": float(before_focus_metrics["loss"]),
                "accuracy": float(before_focus_metrics["accuracy"]),
                "macro_f1": float(before_focus_metrics["macro_f1"]),
            },
            "all": {
                "loss": float(before_all_metrics["loss"]),
                "accuracy": float(before_all_metrics["accuracy"]),
                "macro_f1": float(before_all_metrics["macro_f1"]),
            },
        },
        "after": {
            "focus": {
                "loss": float(after_focus_metrics["loss"]),
                "accuracy": float(after_focus_metrics["accuracy"]),
                "macro_f1": float(after_focus_metrics["macro_f1"]),
            },
            "all": {
                "loss": float(after_all_metrics["loss"]),
                "accuracy": float(after_all_metrics["accuracy"]),
                "macro_f1": float(after_all_metrics["macro_f1"]),
            },
        },
        "delta": {
            "focus_accuracy": float(after_focus_metrics["accuracy"] - before_focus_metrics["accuracy"]),
            "focus_macro_f1": float(after_focus_metrics["macro_f1"] - before_focus_metrics["macro_f1"]),
            "all_accuracy": float(after_all_metrics["accuracy"] - before_all_metrics["accuracy"]),
            "all_macro_f1": float(after_all_metrics["macro_f1"] - before_all_metrics["macro_f1"]),
        },
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "best_monitor_macro_f1": float(best_checkpoint["best_monitor_macro_f1"]),
    }
    save_json(metrics_summary, os.path.join(run_dir, "calibration_metrics_summary.json"))
    print("[INFO] Calibration before/after summary:")
    print(json.dumps(metrics_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
