'''
python test.py \
    --model_path weights_sgfmb5_0.6560_0416/best_RGB_model.pth \
    --image_dir dataset/test2 \
    --output_dir results \
    --use_tta
    ----------
    --color_space RGB   

python test.py \
    --model_path weights_sgfmb5_test2_5/best1_RGB_model.pth \
    --image_dir dataset/test2 \
    --output_dir results \
    --use_tta

python test.py \
    --model_path weights_sgfmb5_0.6560_0416/best_RGB_model.pth \
    --image_dir dataset/train/val/images_1 \
    --output_dir val_results \
    --use_tta
'''


# test.py
import os
import cv2
import torch
import zipfile
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from models.sgfmb5 import MultiModalSegFormerB5

# ===============================
# 配置参数
# ===============================
NUM_CLASSES = 4  # 0:背景, 1:活珊瑚, 2:死珊瑚, 3:白化珊瑚


# ===============================
# 数据集定义（与验证阶段一致）
# ===============================
class TestCoralDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith('.png')
        ])
        self.image_names = [os.path.basename(p) for p in self.image_paths]

        # 与 validate.py 完全一致的预处理
        self.transform = Compose([
            Resize(height=256, width=256),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img, self.image_names[idx]


# ===============================
# 与 validate.py 完全一致的 TTA
# ===============================
def tta_predict(model, images):
    """
    与 validate.py 中完全一致的 TTA 实现：
    - 原图
    - 水平翻转
    - 垂直翻转
    - 旋转90度
    """
    model.eval()

    with torch.no_grad():
        # 原始预测
        orig_pred = model(images)

        # 水平翻转
        flipped_images = torch.flip(images, [-1])
        flipped_pred = model(flipped_images)
        flipped_pred = torch.flip(flipped_pred, [-1])

        # 垂直翻转
        vflipped_images = torch.flip(images, [-2])
        vflipped_pred = model(vflipped_images)
        vflipped_pred = torch.flip(vflipped_pred, [-2])

        # 旋转90度
        rotated_images = torch.rot90(images, k=1, dims=[-2, -1])
        rotated_pred = model(rotated_images)
        rotated_pred = torch.rot90(rotated_pred, k=3, dims=[-2, -1])

        # 融合预测
        final_pred = (orig_pred + flipped_pred +
                      vflipped_pred + rotated_pred) / 4.0

    return final_pred


# ===============================
# 模型加载函数（兼容多种格式）
# ===============================
def load_model(model_path, device,
               use_rgb=True, use_hsv=False, use_lab=False):
    model = MultiModalSegFormerB5(
        num_classes=NUM_CLASSES,
        use_rgb=use_rgb,
        use_hsv=use_hsv,
        use_lab=use_lab,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded direct state_dict.")

    model.eval()
    return model


# ===============================
# 推理函数
# ===============================
def inference(model, loader, device, output_dir, use_tta=False):
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Starting inference | TTA: {use_tta}")
    with torch.no_grad():
        for images, names in tqdm(loader, desc="Inference"):
            images = images.to(device)

            # 是否使用 TTA
            if use_tta:
                outputs = tta_predict(model, images)
            else:
                outputs = model(images)

            bias = torch.tensor(
                [-0.25, 0.0, 0.55, -0.05],
                device=outputs.device
            ).view(1, 4, 1, 1)

            outputs = outputs + bias

            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # 保存预测结果
            for pred, name in zip(preds, names):
                pred_np = pred.cpu().numpy().astype(np.uint8)
                save_path = os.path.join(output_dir, name)
                cv2.imwrite(save_path, pred_np)

    print(f"✅ Predictions saved to: {output_dir}")


# ===============================
# 打包 results 目录
# ===============================
def zip_results(results_dir, zip_name="results.zip"):
    print("📦 Creating results.zip ...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("results", file)
                zipf.write(file_path, arcname)
    print(f"✅ Zip file created: {zip_name}")


# ===============================
# 主函数
# ===============================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Coral Segmentation Test Script")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save prediction masks')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use the same TTA as in validate.py')
    parser.add_argument('--color_space', type=str, default='RGB',
                        choices=['RGB', 'HSV', 'LAB',
                                 'RGB_HSV', 'RGB_LAB', 'RGB_HSV_LAB'],
                        help='Color space configuration')

    args = parser.parse_args()

    # 解析颜色空间
    use_rgb = 'RGB' in args.color_space
    use_hsv = 'HSV' in args.color_space
    use_lab = 'LAB' in args.color_space

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"📌 Using device: {device}")

    # 加载数据
    dataset = TestCoralDataset(args.image_dir)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

    print(f"📂 Number of test images: {len(dataset)}")

    # 加载模型
    model = load_model(args.model_path, device,
                       use_rgb, use_hsv, use_lab)

    # 推理
    inference(model, loader, device,
              args.output_dir, args.use_tta)

    # 打包结果
    zip_results(args.output_dir)


if __name__ == "__main__":
    main()