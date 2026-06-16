"""
python see_over.py \
--images dataset/tra_pri/tra/image/0009.png dataset/tra_pri/tra/label_visual/color_0009.png weights_sgfmb5_test2_4/dirty_data_analysis/dirty_pixel_masks/0009.png \
--alphas 0.4 0.3 0.3 \
--output see/0009.png
"""
import cv2
import numpy as np
import argparse
import os


def load_and_resize(img_path, target_size=None):
    """
    读取并 resize 图片
    """
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    if target_size is not None:
        img = cv2.resize(img, target_size)

    return img


def overlay_images(image_paths, alphas, output_path="overlay.png"):
    """
    多图透明叠加

    Args:
        image_paths: 图片路径列表
        alphas: 每张图透明度
        output_path: 输出路径
    """

    if len(image_paths) not in [2, 3]:
        raise ValueError("仅支持2张或3张图片")

    if len(image_paths) != len(alphas):
        raise ValueError("图片数量和透明度数量必须一致")

    # 读取第一张图
    base_img = cv2.imread(image_paths[0])

    if base_img is None:
        raise ValueError(f"无法读取图片: {image_paths[0]}")

    h, w = base_img.shape[:2]

    # 转 float
    result = np.zeros((h, w, 3), dtype=np.float32)

    # 逐张叠加
    for path, alpha in zip(image_paths, alphas):

        img = load_and_resize(path, (w, h))

        img = img.astype(np.float32)

        result += img * alpha

    # 限制范围
    result = np.clip(result, 0, 255).astype(np.uint8)
    output_dir = os.path.dirname(output_path)
    if output_dir != "":
    	os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, result)

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="输入图片路径（2或3张）"
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        required=True,
        help="每张图片透明度"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="overlay.png",
        help="输出路径"
    )

    args = parser.parse_args()

    overlay_images(
        args.images,
        args.alphas,
        args.output
    )