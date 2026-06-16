# python find_topn_least_diff_predictions.py --root .
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def find_existing(base_dir: Path, stem_candidates: Iterable[str]) -> Optional[Path]:
    for stem in stem_candidates:
        for ext in IMAGE_EXTS:
            candidate = base_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_mask_array(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def resize_like(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 2:
        return np.array(Image.fromarray(mask.astype(np.int32), mode="I").resize(size, Image.NEAREST))
    return np.array(Image.fromarray(mask).resize(size, Image.NEAREST))


def difference_stats(gt: np.ndarray, pred: np.ndarray) -> tuple[int, float]:
    """
    Return (different_pixel_count, different_pixel_ratio).
    Comparison is exact after resizing pred to GT size when needed.
    """
    if gt.ndim == 2:
        if pred.ndim != 2:
            pred = np.squeeze(pred)
            if pred.ndim != 2:
                pred = np.any(pred != pred[:, :, :1], axis=2).astype(np.uint8)
        if gt.shape != pred.shape:
            pred = resize_like(pred, (gt.shape[1], gt.shape[0]))
        mismatch = gt != pred
    else:
        if pred.ndim == 2:
            pred = np.stack([pred] * gt.shape[2], axis=2)
        elif pred.ndim == 3 and pred.shape[2] == 4:
            pred = pred[:, :, :3]
        if gt.shape[:2] != pred.shape[:2]:
            pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
        mismatch = np.any(gt != pred, axis=2)

    diff_pixels = int(mismatch.sum())
    total_pixels = int(mismatch.size)
    diff_ratio = diff_pixels / total_pixels if total_pixels else 0.0
    return diff_pixels, diff_ratio


def resolve_paths(root: Path, stem: str):
    gt_dir = root / "dataset" / "train" / "val" / "masks"
    pred_dir = root / "val_results"
    gt_path = find_existing(gt_dir, [stem])
    pred_path = find_existing(pred_dir, [stem])
    return gt_path, pred_path


def main():
    parser = argparse.ArgumentParser(
        description="Find the prediction masks with the smallest difference from GT."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root that contains work/ and outputs/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=90,
        help="How many least-different predictions to keep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "top40_least_diff.csv",
        help="CSV output path relative to --root unless absolute.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path("outputs") / "top40_least_diff.txt",
        help="Plain-text output path relative to --root unless absolute.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    gt_dir = root / "dataset" / "train" / "val" / "masks"
    pred_dir = root / "val_results"
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=natural_key)
    if not gt_files:
        raise FileNotFoundError(f"No GT masks found in: {gt_dir}")

    records = []
    skipped = 0

    for gt_path in gt_files:
        stem = gt_path.stem
        pred_path = find_existing(pred_dir, [stem])
        if pred_path is None:
            skipped += 1
            continue

        gt_arr = load_mask_array(gt_path)
        pred_arr = load_mask_array(pred_path)
        diff_pixels, diff_ratio = difference_stats(gt_arr, pred_arr)
        records.append(
            {
                "stem": stem,
                "gt_file": gt_path.name,
                "pred_file": pred_path.name,
                "diff_pixels": diff_pixels,
                "diff_ratio": diff_ratio,
            }
        )

    if not records:
        raise RuntimeError("No matched GT/prediction pairs were found.")

    records.sort(key=lambda r: (r["diff_ratio"], r["diff_pixels"], natural_key(Path(r["pred_file"]))))
    top_k = records[: args.top_k]

    out_csv = (root / args.output).resolve()
    out_txt = (root / args.text_output).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "stem", "gt_file", "pred_file", "diff_pixels", "diff_ratio"])
        writer.writeheader()
        for rank, row in enumerate(top_k, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "stem": row["stem"],
                    "gt_file": row["gt_file"],
                    "pred_file": row["pred_file"],
                    "diff_pixels": row["diff_pixels"],
                    "diff_ratio": f"{row['diff_ratio']:.8f}",
                }
            )

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"Top {len(top_k)} least-different prediction files\n")
        f.write(f"Matched pairs: {len(records)}\n")
        f.write(f"Skipped GT files without prediction: {skipped}\n\n")
        for rank, row in enumerate(top_k, start=1):
            f.write(
                f"{rank:02d}. {row['pred_file']} | GT={row['gt_file']} | "
                f"diff_pixels={row['diff_pixels']} | diff_ratio={row['diff_ratio']:.8f}\n"
            )

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote TXT: {out_txt}")
    print(f"Matched pairs: {len(records)}, skipped GT files without prediction: {skipped}")


if __name__ == "__main__":
    main()
