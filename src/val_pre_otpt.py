
'''
python val_pre_otpt.py \
    --model_path weights_sgfmb5_0.6560_0416/best_RGB_model.pth \
    --test_image_dir dataset/test2 \
    --output_root outputs_val \
    --gpu 2 \
    --batch_size 16 \
    --color_space RGB \
    --use_tta
'''
import os
import zipfile
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sgfmb5 import MultiModalSegFormerB5
from utils.mosiac_loader import get_data_loaders

# =========================================================
# TTA（与原验证代码完全一致）
# =========================================================
def tta_predict(model, images):
    """
    与你原代码保持一致：
    1. 原图
    2. 水平翻转
    3. 垂直翻转
    4. 旋转90°
    最终平均 logits
    """

    model.eval()

    with torch.no_grad():
        # 原图
        orig_pred = model(images)

        # 水平翻转
        flipped_images = torch.flip(images, [-1])
        flipped_pred = model(flipped_images)
        flipped_pred = torch.flip(flipped_pred, [-1])

        # 垂直翻转
        vflipped_images = torch.flip(images, [-2])
        vflipped_pred = model(vflipped_images)
        vflipped_pred = torch.flip(vflipped_pred, [-2])

        # 旋转90°
        rotated_images = torch.rot90(images, k=1, dims=[-2, -1])
        rotated_pred = model(rotated_images)
        rotated_pred = torch.rot90(rotated_pred, k=3, dims=[-2, -1])

        # 平均 logits
        final_pred = (
            orig_pred +
            flipped_pred +
            vflipped_pred +
            rotated_pred
        ) / 4.0

    return final_pred


# =========================================================
# zip压缩
# =========================================================
def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                arcname = os.path.relpath(file_path, folder_path)

                zipf.write(file_path, arcname)

    print(f"[INFO] Zip saved to: {zip_path}")


# =========================================================
# 保存概率输出
# =========================================================
def save_probs(
    model,
    loader,
    save_dir,
    device,
    use_tta=False
):
    """
    输出：
        每张图 -> 4 x 256 x 256 的 float32 概率

    类别顺序固定：
        0,1,2,3

    保存格式：
        xxx.npy
    """

    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():

        for batch in tqdm(loader):

            # -------------------------------------------------
            # 兼容不同 dataloader 返回格式
            # -------------------------------------------------
            if len(batch) == 3:
                images, _, image_ids = batch
            else:
                raise ValueError("Unsupported dataloader output format")

            images = images.to(device)

            # -------------------------------------------------
            # 推理
            # -------------------------------------------------
            if use_tta:
                logits = tta_predict(model, images)
            else:
                logits = model(images)

            # -------------------------------------------------
            # softmax 概率
            # shape:
            #   B x 4 x 256 x 256
            # -------------------------------------------------
            probs = F.softmax(logits, dim=1)

            probs = probs.cpu().numpy().astype(np.float32)

            # -------------------------------------------------
            # 保存每张图
            # -------------------------------------------------
            for i in range(probs.shape[0]):

                prob = probs[i]

                # 确保 shape 正确
                assert prob.shape[0] == 4

                image_id = image_ids[i]

                # 去后缀
                image_id = os.path.splitext(image_id)[0]

                save_path = os.path.join(
                    save_dir,
                    f"{image_id}.npy"
                )

                np.save(save_path, prob)


save_probs.global_idx = 0


# =========================================================
# 主函数
# =========================================================
def main(args):

    DEVICE = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )

    NUM_CLASSES = 4

    print("=" * 60)
    print(f"Device      : {DEVICE}")
    print(f"Use TTA     : {args.use_tta}")
    print(f"Color Space : {args.color_space}")
    print("=" * 60)

    # =====================================================
    # 颜色空间解析
    # =====================================================
    use_rgb = 'RGB' in args.color_space
    use_hsv = 'HSV' in args.color_space
    use_lab = 'LAB' in args.color_space

    # =====================================================
    # 初始化模型
    # =====================================================
    model = MultiModalSegFormerB5(
        num_classes=NUM_CLASSES,
        use_rgb=use_rgb,
        use_hsv=use_hsv,
        use_lab=use_lab,
        pretrained=False
    ).to(DEVICE)

    # =====================================================
    # 加载权重
    # =====================================================
    print(f"[INFO] Loading model: {args.model_path}")

    checkpoint = torch.load(
        args.model_path,
        map_location=DEVICE
    )

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[INFO] Loaded checkpoint format")

    except:
        model.load_state_dict(checkpoint)
        print("[INFO] Loaded raw state_dict")

    model.eval()

    # =====================================================
    # VAL Loader
    # =====================================================
    _, val_loader = get_data_loaders(
        train_image_dir='dataset/tra_pri/tra/image',
        train_mask_dir='dataset/tra_pri/tra/label',
        val_image_dir='dataset/tra_pri/val/image',
        val_mask_dir='dataset/tra_pri/val/label',
        batch_size=args.batch_size,
    )

    # =====================================================
    # TEST Loader
    # =====================================================
    # 你需要保证这里的 test loader
    # 返回:
    #   images, image_ids
    #
    # image_ids:
    #   例如:
    #   0001.png
    #
    # -----------------------------------------------------
    from utils.test_loader import TestDataset

    test_dataset = TestDataset(
        image_dir=args.test_image_dir,
        use_rgb=use_rgb,
        use_hsv=use_hsv,
        use_lab=use_lab
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =====================================================
    # 输出目录
    # =====================================================
    os.makedirs(args.output_root, exist_ok=True)

    val_save_dir = os.path.join(args.output_root, "val_probs")
    test_save_dir = os.path.join(args.output_root, "test_probs")

    # =====================================================
    # 保存 val probs
    # =====================================================
    print("\n[INFO] Saving val probabilities...")

    save_probs(
        model=model,
        loader=val_loader,
        save_dir=val_save_dir,
        device=DEVICE,
        use_tta=args.use_tta
    )

    # =====================================================
    # 保存 test probs
    # =====================================================
    print("\n[INFO] Saving test probabilities...")

    save_probs(
        model=model,
        loader=test_loader,
        save_dir=test_save_dir,
        device=DEVICE,
        use_tta=args.use_tta
    )

    # =====================================================
    # 压缩
    # =====================================================
    val_zip_path = os.path.join(
        args.output_root,
        "val_probs.zip"
    )

    test_zip_path = os.path.join(
        args.output_root,
        "test_probs.zip"
    )

    print("\n[INFO] Zipping val probs...")
    zip_folder(val_save_dir, val_zip_path)

    print("\n[INFO] Zipping test probs...")
    zip_folder(test_save_dir, test_zip_path)

    print("\n[INFO] Done!")


# =========================================================
# 启动
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # -----------------------------------------------------
    # 模型
    # -----------------------------------------------------
    parser.add_argument(
        '--model_path',
        type=str,
        required=True
    )

    # -----------------------------------------------------
    # testB 路径
    # -----------------------------------------------------
    parser.add_argument(
        '--test_image_dir',
        type=str,
        required=True
    )

    # -----------------------------------------------------
    # 输出目录
    # -----------------------------------------------------
    parser.add_argument(
        '--output_root',
        type=str,
        default='prob_outputs'
    )

    # -----------------------------------------------------
    # batch size
    # -----------------------------------------------------
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )

    # -----------------------------------------------------
    # GPU
    # -----------------------------------------------------
    parser.add_argument(
        '--gpu',
        type=int,
        default=0
    )

    # -----------------------------------------------------
    # TTA
    # -----------------------------------------------------
    parser.add_argument(
        '--use_tta',
        action='store_true',
        default=False
    )

    # -----------------------------------------------------
    # 颜色空间
    # -----------------------------------------------------
    parser.add_argument(
        '--color_space',
        type=str,
        default='RGB',
        choices=[
            'RGB',
            'HSV',
            'LAB',
            'RGB_HSV',
            'RGB_LAB',
            'RGB_HSV_LAB'
        ]
    )

    args = parser.parse_args()

    main(args)