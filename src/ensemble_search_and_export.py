#!/usr/bin/env python
from __future__ import annotations

"""Search ensemble weights on validation probabilities and export test PNGs(模型融合)."""

"""
python ensemble_search_and_export.py \
  --val_probs outputs/val_probs_xhr outputs/val_probs_sjm outputs/val_probs_cly \
  --test_probs outputs/test_probs_xhr outputs/test_probs_sjm outputs/test_probs_cly \
  --val_labels dataset/tra_pri/val/label \
  --out_dir fused_output_class_specific \
"""
import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse multiple segmentation probability directories, search global "
            "ensemble weights on a validation set, and export test predictions."
        )
    )
    parser.add_argument(
        "--val_probs",
        nargs="+",
        required=True,
        help="Validation probability directories, one per model.",
    )
    parser.add_argument(
        "--test_probs",
        nargs="+",
        required=True,
        help="Test probability directories, one per model.",
    )
    parser.add_argument(
        "--val_labels",
        required=True,
        help="Validation label PNG directory. Each file stem must match val_probs.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for JSON summaries and fused test PNGs.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of segmentation classes. Default: 4.",
    )
    parser.add_argument(
        "--coarse_step",
        type=float,
        default=0.1,
        help="Coarse search step on the simplex. Default: 0.1.",
    )
    parser.add_argument(
        "--fine_step",
        type=float,
        default=0.05,
        help="Fine search step around coarse optimum. Default: 0.05.",
    )
    parser.add_argument(
        "--fine_radius",
        type=float,
        default=0.1,
        help="Local search radius around the coarse optimum. Default: 0.1.",
    )
    parser.add_argument(
        "--batch_candidates",
        type=int,
        default=16,
        help="How many weight candidates to score together. Default: 16.",
    )
    return parser.parse_args()


def list_npy_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(directory.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {directory}")
    return {path.stem: path for path in files}


def list_label_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Label directory not found: {directory}")
    files = sorted(directory.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png label files found in: {directory}")
    return {path.stem: path for path in files}


def find_common_stems(path_maps: Sequence[dict[str, Path]], kind: str) -> list[str]:
    common = set(path_maps[0])
    for path_map in path_maps[1:]:
        common &= set(path_map)
    if not common:
        raise ValueError(f"No common sample stems found across {kind} directories.")
    return sorted(common)


def load_prob_map(prob_path: Path, num_classes: int) -> np.ndarray:
    array = np.load(prob_path)
    if array.ndim != 3:
        raise ValueError(
            f"Probability file must be 3D, got shape {array.shape} in {prob_path}"
        )

    if array.shape[0] == num_classes:
        prob_map = array
    elif array.shape[-1] == num_classes:
        prob_map = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(
            f"Could not infer channel axis for {prob_path}, got shape {array.shape}"
        )

    if prob_map.shape[0] != num_classes:
        raise ValueError(
            f"Expected {num_classes} classes in {prob_path}, got {prob_map.shape[0]}"
        )
    return prob_map.astype(np.float32, copy=False)


def load_label_map(label_path: Path, num_classes: int) -> np.ndarray:
    label = np.array(Image.open(label_path), dtype=np.uint8)
    if label.ndim != 2:
        raise ValueError(f"Label file must be single-channel: {label_path}")
    unique_values = np.unique(label)
    invalid = unique_values[(unique_values < 0) | (unique_values >= num_classes)]
    if invalid.size > 0:
        raise ValueError(
            f"Label {label_path} contains invalid class ids: {invalid.tolist()}"
        )
    return label


def generate_simplex_weights(num_models: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"Step {step} must evenly divide 1.0.")

    candidates: list[np.ndarray] = []
    current = [0] * num_models

    def backtrack(index: int, remaining: int) -> None:
        if index == num_models - 1:
            current[index] = remaining
            candidates.append(np.array(current, dtype=np.float32) * step)
            return
        for value in range(remaining + 1):
            current[index] = value
            backtrack(index + 1, remaining - value)

    backtrack(0, units)
    return candidates


def generate_local_simplex_weights(
    center: np.ndarray,
    step: float,
    radius: float,
) -> list[np.ndarray]:
    num_models = len(center)
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"Step {step} must evenly divide 1.0.")

    lower_bounds = np.maximum(center - radius, 0.0)
    upper_bounds = np.minimum(center + radius, 1.0)

    lower_units = np.ceil(lower_bounds / step - 1e-8).astype(int)
    upper_units = np.floor(upper_bounds / step + 1e-8).astype(int)

    candidates: list[np.ndarray] = []
    current = [0] * num_models

    def backtrack(index: int, remaining: int) -> None:
        if index == num_models - 1:
            if lower_units[index] <= remaining <= upper_units[index]:
                current[index] = remaining
                candidates.append(np.array(current, dtype=np.float32) * step)
            return

        min_value = max(lower_units[index], 0)
        max_value = min(upper_units[index], remaining)
        for value in range(min_value, max_value + 1):
            current[index] = value
            backtrack(index + 1, remaining - value)

    backtrack(0, units)
    if not candidates:
        candidates.append(center.astype(np.float32, copy=True))
    return candidates


def deduplicate_weights(weights: Iterable[np.ndarray]) -> list[np.ndarray]:
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for weight in weights:
        key = tuple(np.round(weight.astype(np.float64), 8).tolist())
        unique[key] = weight.astype(np.float32, copy=False)
    return list(unique.values())


def confusion_to_metrics(confusion: np.ndarray) -> tuple[list[float | None], float]:
    ious: list[float | None] = []
    valid_ious: list[float] = []
    for class_idx in range(confusion.shape[0]):
        tp = float(confusion[class_idx, class_idx])
        fp = float(confusion[:, class_idx].sum() - tp)
        fn = float(confusion[class_idx, :].sum() - tp)
        union = tp + fp + fn
        if union == 0:
            ious.append(None)
            continue
        iou = tp / union
        ious.append(iou)
        valid_ious.append(iou)
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return ious, miou


def evaluate_candidates(
    val_stems: Sequence[str],
    val_prob_maps: Sequence[dict[str, Path]],
    label_map: dict[str, Path],
    weight_candidates: Sequence[np.ndarray],
    num_classes: int,
    batch_candidates: int,
) -> list[dict[str, object]]:
    confusions = np.zeros(
        (len(weight_candidates), num_classes, num_classes), dtype=np.int64
    )

    for sample_index, stem in enumerate(val_stems, start=1):
        label = load_label_map(label_map[stem], num_classes)
        prob_stack = np.stack(
            [load_prob_map(prob_map[stem], num_classes) for prob_map in val_prob_maps],
            axis=0,
        )

        height, width = label.shape
        if tuple(prob_stack.shape[-2:]) != (height, width):
            raise ValueError(
                f"Shape mismatch for sample {stem}: label {label.shape}, "
                f"probability {tuple(prob_stack.shape[-2:])}"
            )

        label_flat = label.reshape(-1).astype(np.int64, copy=False)

        for start in range(0, len(weight_candidates), batch_candidates):
            end = min(start + batch_candidates, len(weight_candidates))
            weight_batch = np.stack(weight_candidates[start:end], axis=0)
            fused_batch = np.tensordot(weight_batch, prob_stack, axes=(1, 0))
            pred_batch = fused_batch.argmax(axis=1).reshape(end - start, -1)

            for local_idx, pred_flat in enumerate(pred_batch):
                bincount = np.bincount(
                    label_flat * num_classes + pred_flat.astype(np.int64, copy=False),
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)
                confusions[start + local_idx] += bincount

        if sample_index % 20 == 0 or sample_index == len(val_stems):
            print(f"[val] processed {sample_index}/{len(val_stems)} samples")

    results: list[dict[str, object]] = []
    for weight, confusion in zip(weight_candidates, confusions):
        ious, miou = confusion_to_metrics(confusion)
        results.append(
            {
                "weights": [float(x) for x in weight.tolist()],
                "ious": ious,
                "miou": miou,
            }
        )
    return results


def evaluate_weight_matrix_candidates(
    val_stems: Sequence[str],
    val_prob_maps: Sequence[dict[str, Path]],
    label_map: dict[str, Path],
    weight_matrices: Sequence[np.ndarray],
    num_classes: int,
    batch_candidates: int,
) -> list[dict[str, object]]:
    confusions = np.zeros(
        (len(weight_matrices), num_classes, num_classes), dtype=np.int64
    )

    for sample_index, stem in enumerate(val_stems, start=1):
        label = load_label_map(label_map[stem], num_classes)
        prob_stack = np.stack(
            [load_prob_map(prob_map[stem], num_classes) for prob_map in val_prob_maps],
            axis=0,
        )

        height, width = label.shape
        if tuple(prob_stack.shape[-2:]) != (height, width):
            raise ValueError(
                f"Shape mismatch for sample {stem}: label {label.shape}, "
                f"probability {tuple(prob_stack.shape[-2:])}"
            )

        label_flat = label.reshape(-1).astype(np.int64, copy=False)

        for start in range(0, len(weight_matrices), batch_candidates):
            end = min(start + batch_candidates, len(weight_matrices))
            weight_batch = np.stack(weight_matrices[start:end], axis=0)
            fused_batch = (
                weight_batch[:, :, :, None, None] * prob_stack[None, :, :, :, :]
            ).sum(axis=1)
            pred_batch = fused_batch.argmax(axis=1).reshape(end - start, -1)

            for local_idx, pred_flat in enumerate(pred_batch):
                bincount = np.bincount(
                    label_flat * num_classes + pred_flat.astype(np.int64, copy=False),
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)
                confusions[start + local_idx] += bincount

        if sample_index % 20 == 0 or sample_index == len(val_stems):
            print(
                f"[val-class] processed {sample_index}/{len(val_stems)} samples"
            )

    results: list[dict[str, object]] = []
    for weight_matrix, confusion in zip(weight_matrices, confusions):
        ious, miou = confusion_to_metrics(confusion)
        results.append(
            {
                "weight_matrix": weight_matrix.tolist(),
                "ious": ious,
                "miou": miou,
            }
        )
    return results


def search_class_specific_weights(
    val_stems: Sequence[str],
    val_prob_maps: Sequence[dict[str, Path]],
    label_map: dict[str, Path],
    init_global_weight: np.ndarray,
    num_classes: int,
    batch_candidates: int,
    coarse_step: float,
    fine_step: float,
    fine_radius: float,
    class_indices: Sequence[int],
    rounds: int = 2,
) -> dict[str, object]:
    num_models = len(init_global_weight)
    current = np.repeat(
        init_global_weight[:, None], num_classes, axis=1
    ).astype(np.float32)

    base_result = evaluate_weight_matrix_candidates(
        val_stems=val_stems,
        val_prob_maps=val_prob_maps,
        label_map=label_map,
        weight_matrices=[current],
        num_classes=num_classes,
        batch_candidates=batch_candidates,
    )[0]
    best_miou = float(base_result["miou"])

    for round_index in range(rounds):
        improved = False
        print(
            f"Searching class-specific weights, round {round_index + 1}/{rounds}..."
        )
        for class_idx in class_indices:
            print(f"  Optimizing class {class_idx}...")
            coarse_simplex = generate_simplex_weights(num_models, coarse_step)
            coarse_candidates = []
            for simplex_weight in coarse_simplex:
                candidate = current.copy()
                candidate[:, class_idx] = simplex_weight
                coarse_candidates.append(candidate)

            coarse_results = evaluate_weight_matrix_candidates(
                val_stems=val_stems,
                val_prob_maps=val_prob_maps,
                label_map=label_map,
                weight_matrices=coarse_candidates,
                num_classes=num_classes,
                batch_candidates=batch_candidates,
            )
            coarse_best = max(coarse_results, key=lambda item: float(item["miou"]))
            coarse_best_matrix = np.array(
                coarse_best["weight_matrix"], dtype=np.float32
            )

            fine_simplex = generate_local_simplex_weights(
                center=coarse_best_matrix[:, class_idx],
                step=fine_step,
                radius=fine_radius,
            )
            fine_candidates = []
            for simplex_weight in fine_simplex:
                candidate = coarse_best_matrix.copy()
                candidate[:, class_idx] = simplex_weight
                fine_candidates.append(candidate)

            fine_results = evaluate_weight_matrix_candidates(
                val_stems=val_stems,
                val_prob_maps=val_prob_maps,
                label_map=label_map,
                weight_matrices=fine_candidates,
                num_classes=num_classes,
                batch_candidates=batch_candidates,
            )
            fine_best = max(fine_results, key=lambda item: float(item["miou"]))

            if float(fine_best["miou"]) > best_miou:
                current = np.array(fine_best["weight_matrix"], dtype=np.float32)
                best_miou = float(fine_best["miou"])
                improved = True
                print(
                    f"  Class {class_idx} improved validation mIoU to {best_miou:.6f}"
                )

        if not improved:
            print("No further class-specific improvement found, stopping early.")
            break

    final_result = evaluate_weight_matrix_candidates(
        val_stems=val_stems,
        val_prob_maps=val_prob_maps,
        label_map=label_map,
        weight_matrices=[current],
        num_classes=num_classes,
        batch_candidates=batch_candidates,
    )[0]

    return {
        "weight_matrix": current,
        "miou": float(final_result["miou"]),
        "ious": final_result["ious"],
    }


def build_sample_maps(directories: Sequence[Path]) -> tuple[list[dict[str, Path]], list[str]]:
    path_maps = [list_npy_map(directory) for directory in directories]
    stems = find_common_stems(path_maps, "probability")
    return path_maps, stems


def build_validation_inputs(
    val_prob_dirs: Sequence[Path],
    label_dir: Path,
) -> tuple[list[dict[str, Path]], dict[str, Path], list[str]]:
    val_prob_maps, val_stems = build_sample_maps(val_prob_dirs)
    label_map = list_label_map(label_dir)
    available_stems = sorted(set(val_stems) & set(label_map))
    if not available_stems:
        raise ValueError("No common stems found between validation probabilities and labels.")
    return val_prob_maps, label_map, available_stems


def build_test_inputs(test_prob_dirs: Sequence[Path]) -> tuple[list[dict[str, Path]], list[str]]:
    return build_sample_maps(test_prob_dirs)


def format_metrics(result: dict[str, object]) -> dict[str, object]:
    ious = result["ious"]
    formatted = {"miou": float(result["miou"])}
    for class_idx, iou in enumerate(ious):
        formatted[f"iou_{class_idx}"] = None if iou is None else float(iou)
    return formatted


def infer_model_names(prob_dirs: Sequence[Path]) -> list[str]:
    return [path.name for path in prob_dirs]


def export_test_predictions(
    test_stems: Sequence[str],
    test_prob_maps: Sequence[dict[str, Path]],
    weights: np.ndarray,
    out_dir: Path,
    num_classes: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_class_specific = weights.ndim == 2
    for sample_index, stem in enumerate(test_stems, start=1):
        prob_stack = np.stack(
            [load_prob_map(prob_map[stem], num_classes) for prob_map in test_prob_maps],
            axis=0,
        )
        if use_class_specific:
            fused = (weights[:, :, None, None] * prob_stack).sum(axis=0)
        else:
            fused = np.tensordot(weights, prob_stack, axes=(0, 0))
        pred = fused.argmax(axis=0).astype(np.uint8)
        Image.fromarray(pred, mode="L").save(out_dir / f"{stem}.png")

        if sample_index % 50 == 0 or sample_index == len(test_stems):
            print(f"[test] exported {sample_index}/{len(test_stems)} samples")


def main() -> None:
    args = parse_args()

    val_prob_dirs = [Path(path) for path in args.val_probs]
    test_prob_dirs = [Path(path) for path in args.test_probs]
    label_dir = Path(args.val_labels)
    out_dir = Path(args.out_dir)

    if len(val_prob_dirs) != len(test_prob_dirs):
        raise ValueError(
            f"--val_probs count ({len(val_prob_dirs)}) must match --test_probs count "
            f"({len(test_prob_dirs)})."
        )

    num_models = len(val_prob_dirs)
    if num_models < 1:
        raise ValueError("At least one model directory is required.")

    print("Preparing validation inputs...")
    val_prob_maps, label_map, val_stems = build_validation_inputs(val_prob_dirs, label_dir)
    print(f"Found {len(val_stems)} validation samples shared by all models and labels.")

    print("Preparing test inputs...")
    test_prob_maps, test_stems = build_test_inputs(test_prob_dirs)
    print(f"Found {len(test_stems)} test samples shared by all models.")

    model_names = infer_model_names(val_prob_dirs)
    identity_weights = [np.eye(num_models, dtype=np.float32)[i] for i in range(num_models)]
    equal_weight = np.full(num_models, 1.0 / num_models, dtype=np.float32)
    coarse_candidates = generate_simplex_weights(num_models, args.coarse_step)
    coarse_eval_candidates = deduplicate_weights(identity_weights + [equal_weight] + coarse_candidates)

    print(
        f"Running coarse search with {len(coarse_eval_candidates)} candidates "
        f"(step={args.coarse_step})..."
    )
    coarse_results = evaluate_candidates(
        val_stems=val_stems,
        val_prob_maps=val_prob_maps,
        label_map=label_map,
        weight_candidates=coarse_eval_candidates,
        num_classes=args.num_classes,
        batch_candidates=args.batch_candidates,
    )

    result_by_key = {
        tuple(np.round(np.array(result["weights"], dtype=np.float64), 8).tolist()): result
        for result in coarse_results
    }
    single_results = []
    for model_name, weight in zip(model_names, identity_weights):
        key = tuple(np.round(weight.astype(np.float64), 8).tolist())
        single_results.append((model_name, result_by_key[key]))

    equal_key = tuple(np.round(equal_weight.astype(np.float64), 8).tolist())
    equal_result = result_by_key[equal_key]

    coarse_best = max(coarse_results, key=lambda item: float(item["miou"]))
    coarse_best_weight = np.array(coarse_best["weights"], dtype=np.float32)

    fine_candidates = deduplicate_weights(
        [coarse_best_weight]
        + generate_local_simplex_weights(
            center=coarse_best_weight,
            step=args.fine_step,
            radius=args.fine_radius,
        )
    )
    print(
        f"Running fine search with {len(fine_candidates)} candidates "
        f"(step={args.fine_step}, radius={args.fine_radius})..."
    )
    fine_results = evaluate_candidates(
        val_stems=val_stems,
        val_prob_maps=val_prob_maps,
        label_map=label_map,
        weight_candidates=fine_candidates,
        num_classes=args.num_classes,
        batch_candidates=args.batch_candidates,
    )
    best_result = max(fine_results, key=lambda item: float(item["miou"]))
    best_weight = np.array(best_result["weights"], dtype=np.float32)

    print("Searching class-specific weights for foreground classes...")
    foreground_classes = [class_idx for class_idx in range(1, args.num_classes)]
    class_specific_result = search_class_specific_weights(
        val_stems=val_stems,
        val_prob_maps=val_prob_maps,
        label_map=label_map,
        init_global_weight=best_weight,
        num_classes=args.num_classes,
        batch_candidates=args.batch_candidates,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        fine_radius=args.fine_radius,
        class_indices=foreground_classes,
        rounds=2,
    )
    class_specific_weight_matrix = np.array(
        class_specific_result["weight_matrix"], dtype=np.float32
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = {
        "num_classes": args.num_classes,
        "num_models": num_models,
        "num_val_samples": len(val_stems),
        "num_test_samples": len(test_stems),
        "model_names": model_names,
        "single_models": {
            model_name: format_metrics(result) for model_name, result in single_results
        },
        "equal_weight_ensemble": {
            "weights": [float(x) for x in equal_weight.tolist()],
            **format_metrics(equal_result),
        },
        "best_weight_ensemble": {
            "weights": [float(x) for x in best_weight.tolist()],
            **format_metrics(best_result),
        },
        "class_specific_ensemble": {
            "weight_matrix": class_specific_weight_matrix.tolist(),
            "optimized_classes": foreground_classes,
            **format_metrics(class_specific_result),
        },
    }
    best_weights_json = {
        "model_names": model_names,
        "best_weights": [float(x) for x in best_weight.tolist()],
        "best_miou": float(best_result["miou"]),
        "class_specific_weight_matrix": class_specific_weight_matrix.tolist(),
        "class_specific_best_miou": float(class_specific_result["miou"]),
        "optimized_classes": foreground_classes,
        "coarse_step": args.coarse_step,
        "fine_step": args.fine_step,
        "fine_radius": args.fine_radius,
    }

    with (out_dir / "val_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_json, file, indent=2, ensure_ascii=False)

    with (out_dir / "best_weights.json").open("w", encoding="utf-8") as file:
        json.dump(best_weights_json, file, indent=2, ensure_ascii=False)

    print("Exporting fused test predictions...")
    export_test_predictions(
        test_stems=test_stems,
        test_prob_maps=test_prob_maps,
        weights=class_specific_weight_matrix,
        out_dir=out_dir / "test_pred",
        num_classes=args.num_classes,
    )

    print("Done.")
    print(f"Best global weights: {best_weights_json['best_weights']}")
    print(f"Best global validation mIoU: {best_weights_json['best_miou']:.6f}")
    print(
        "Best class-specific validation mIoU: "
        f"{best_weights_json['class_specific_best_miou']:.6f}"
    )
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
