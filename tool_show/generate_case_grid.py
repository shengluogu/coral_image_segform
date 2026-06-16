'''
自动读取 work/CDNet/dataset/train/val/images_1
配对 masks、masks_visual、val_results、val_results_visual
生成你要的 2x2 拼图：Image / GT / Pred / Error
按顺序保存为 outputs/case_grid/case_01.png, outputs/case_grid/case_02.png ...
'''
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps


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


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_array(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def resize_for_panel(img: Image.Image, size: tuple[int, int], is_mask: bool) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BICUBIC
    return ImageOps.contain(img, size, method=resample)


def center_on_canvas(img: Image.Image, size: tuple[int, int], fill=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", size, fill)
    left = (size[0] - img.width) // 2
    top = (size[1] - img.height) // 2
    canvas.paste(img, (left, top))
    return canvas


def panel_with_title(
    content: Image.Image,
    title: str,
    size: tuple[int, int],
    is_mask: bool = False,
    title_bar_h: int = 34,
    border: int = 2,
) -> Image.Image:
    resized = resize_for_panel(content, (size[0] - border * 2, size[1] - title_bar_h - border * 2), is_mask)
    body = center_on_canvas(resized, (size[0] - border * 2, size[1] - title_bar_h - border * 2))

    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(35, 35, 35), width=border)
    draw.rectangle([border, border, size[0] - border - 1, title_bar_h], fill=(240, 240, 240))
    draw.line([(border, title_bar_h), (size[0] - border - 1, title_bar_h)], fill=(35, 35, 35), width=1)

    font = ImageFont_safe()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = max(border + 8, (size[0] - tw) // 2)
    ty = max(border + 6, (title_bar_h - th) // 2 - 1)
    draw.text((tx, ty), title, fill=(20, 20, 20), font=font)

    panel.paste(body, (border, title_bar_h + border))
    return panel


def ImageFont_safe():
    try:
        from PIL import ImageFont

        return ImageFont.load_default()
    except Exception:
        return None


def is_binary_like(mask: np.ndarray) -> bool:
    if mask.ndim == 3:
        return False
    values = np.unique(mask)
    if len(values) <= 2:
        return True
    normalized = set(int(v) for v in values.tolist())
    return normalized.issubset({0, 1, 255})


def make_error_panel(
    gt: np.ndarray,
    pred: np.ndarray,
    original: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    target = original.convert("RGB").resize(size, Image.BICUBIC).copy()
    arr = np.array(target)

    if gt.ndim == 2:
        gt_img = Image.fromarray(gt.astype(np.uint8)).resize(size, Image.NEAREST)
        pred_img = Image.fromarray(pred.astype(np.uint8)).resize(size, Image.NEAREST)
        gt_arr = np.array(gt_img)
        pred_arr = np.array(pred_img)
        mismatch = gt_arr != pred_arr
    else:
        gt_img = Image.fromarray(gt).resize(size, Image.NEAREST).convert("RGB")
        pred_img = Image.fromarray(pred).resize(size, Image.NEAREST).convert("RGB")
        gt_arr = np.array(gt_img)
        pred_arr = np.array(pred_img)
        mismatch = np.any(gt_arr != pred_arr, axis=2)

    arr[:] = (24, 24, 24)
    arr[mismatch] = (220, 50, 50)
    return Image.fromarray(arr)

def make_visual_panel(
    image_path: Path,
    raw_mask_path: Path,
    visual_path: Optional[Path],
    fallback_title: str,
    size: tuple[int, int],
    is_mask: bool,
) -> Image.Image:
    if visual_path and visual_path.exists():
        content = load_rgb(visual_path)
    elif raw_mask_path.exists():
        raw = Image.open(raw_mask_path)
        content = raw.convert("RGB") if raw.mode != "RGB" else raw
    else:
        content = load_rgb(image_path)
    return panel_with_title(content, fallback_title, size, is_mask=is_mask)


def build_case_grid(
    image_path: Path,
    gt_mask_path: Path,
    pred_mask_path: Path,
    gt_vis_path: Optional[Path],
    pred_vis_path: Optional[Path],
    output_path: Path,
    labels: tuple[str, str, str, str] = ("Image", "GT", "Pred", "Error"),
) -> None:
    original = load_rgb(image_path)
    gt_arr = load_mask_array(gt_mask_path)
    pred_arr = load_mask_array(pred_mask_path)

    target_size = original.size
    panel_size = (target_size[0], target_size[1] + 34 + 4)

    image_panel = panel_with_title(original, labels[0], panel_size, is_mask=False)
    gt_panel = make_visual_panel(image_path, gt_mask_path, gt_vis_path, labels[1], panel_size, is_mask=True)
    pred_panel = make_visual_panel(image_path, pred_mask_path, pred_vis_path, labels[2], panel_size, is_mask=True)
    error_content = make_error_panel(gt_arr, pred_arr, original, target_size)
    error_panel = panel_with_title(error_content, labels[3], panel_size, is_mask=False)

    gap = 2
    canvas_w = panel_size[0] * 2 + gap
    canvas_h = panel_size[1] * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(image_panel, (0, 0))
    canvas.paste(gt_panel, (panel_size[0] + gap, 0))
    canvas.paste(pred_panel, (0, panel_size[1] + gap))
    canvas.paste(error_panel, (panel_size[0] + gap, panel_size[1] + gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def resolve_paths(root: Path, stem: str):
    image_dir = root /  "dataset" / "train" / "val" / "images_1"
    gt_mask_dir = root /  "dataset" / "train" / "val" / "masks"
    gt_vis_dir = root /  "dataset" / "train" / "val" / "masks_visual"
    pred_mask_dir = root / "val_results"
    pred_vis_dir = root / "val_results_visual"

    image_path = find_existing(image_dir, [stem])
    gt_mask_path = find_existing(gt_mask_dir, [stem])
    pred_mask_path = find_existing(pred_mask_dir, [stem])
    gt_vis_path = find_existing(gt_vis_dir, [f"color_{stem}", stem])
    pred_vis_path = find_existing(pred_vis_dir, [f"color_{stem}", stem])

    return image_path, gt_mask_path, pred_mask_path, gt_vis_path, pred_vis_path


def main():
    parser = argparse.ArgumentParser(
        description="Build 2x2 comparison grids: Image / GT / Pred / Error."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root that contains work/ and outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/case_grid"),
        help="Directory for case_grid/case_XX.png files, relative to --root by default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to export.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    image_dir = root / "dataset" / "train" / "val" / "images_1"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )
    if args.limit is not None:
        image_files = image_files[: args.limit]

    if not image_files:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for idx, image_path in enumerate(image_files, start=1):
        stem = image_path.stem
        image_path2, gt_mask_path, pred_mask_path, gt_vis_path, pred_vis_path = resolve_paths(root, stem)

        if not image_path2 or not gt_mask_path or not pred_mask_path:
            skipped += 1
            print(f"[skip] {stem}: missing one or more required files")
            continue

        out_name = f"{stem}.png"
        out_path = output_dir / out_name
        build_case_grid(
            image_path=image_path2,
            gt_mask_path=gt_mask_path,
            pred_mask_path=pred_mask_path,
            gt_vis_path=gt_vis_path,
            pred_vis_path=pred_vis_path,
            output_path=out_path,
        )
        written += 1
        print(f"[ok] {out_name}")

    print(f"Done. written={written}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
