'''
python validate.py --model_path weights/checkpoint_RGB_epoch50.pth --use_tta
python validate.py --model_path weights_sgfmb5_0.6560_0416/best_RGB_model.pth --use_tta
python validate.py --model_path weights_sgfmb5/best_RGB_model.pth --use_tta
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from models.causal_deeplabv3plus import MultiModalDeepLabV3Plus
from models.sgfmb5 import MultiModalSegFormerB5
from utils.mosiac_loader import get_data_loaders
from utils.Loss import LovaszSoftmaxLoss, CrossEntropyLoss, DiceLoss, BoundaryLoss, CVELoss
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from logging.handlers import RotatingFileHandler
import time
from tqdm import tqdm
def compute_detailed_metrics(hist):
    """
    从混淆矩阵计算：
    - per-class accuracy (recall)
    - per-class precision
    - confusion probability matrix P(pred=j | gt=i)
    """
    hist = hist.float()

    # TP
    diag = torch.diag(hist)

    # GT 总数 (row sum)
    gt_sum = hist.sum(1)

    # Pred 总数 (col sum)
    pred_sum = hist.sum(0)

    # ===== 1️⃣ Per-class Accuracy (Recall) =====
    acc_per_class = diag / (gt_sum + 1e-10)

    # ===== 2️⃣ Precision =====
    precision_per_class = diag / (pred_sum + 1e-10)

    # ===== 3️⃣ Confusion Probability =====
    # 行归一化：P(pred=j | gt=i)
    confusion_prob = hist / (gt_sum.unsqueeze(1) + 1e-10)

    return (
        acc_per_class.cpu().numpy(),
        precision_per_class.cpu().numpy(),
        confusion_prob.cpu().numpy()
    )
def fast_hist(pred, target, num_classes, ignore_index=255):
    """
    计算混淆矩阵
    """
    pred = torch.argmax(pred, dim=1)
    
    pred = pred.view(-1)
    target = target.view(-1)

    # 过滤忽略像素
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    hist = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return hist


def compute_iou_from_hist(hist):
    """
    根据混淆矩阵计算每类 IoU 和 mIoU
    """
    hist = hist.float()
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou_per_class = intersection / (union + 1e-10)
    miou = iou_per_class.mean()

    return miou.item(), iou_per_class.cpu().numpy()
def setup_logger(log_dir='logs', log_name='validation'):
    """配置日志记录器，同时输出到文件和控制台"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger('ValidationLogger')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器（带轮转，防止文件过大）
    log_file = os.path.join(log_dir, f'{log_name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024, # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def tta_predict(model, images, device):
    """
    测试时增强预测
    对图像进行多种变换后预测，然后融合结果
    """
    model.eval()
    
    # 原始图像预测
    with torch.no_grad():
        orig_pred = model(images)
    
    # 水平翻转
    flipped_images = torch.flip(images, [-1])
    with torch.no_grad():
        flipped_pred = model(flipped_images)
    # 将翻转后的预测结果再翻转回来
    flipped_pred = torch.flip(flipped_pred, [-1])
    
    # 垂直翻转
    vflipped_images = torch.flip(images, [-2])
    with torch.no_grad():
        vflipped_pred = model(vflipped_images)
    # 将翻转后的预测结果再翻转回来
    vflipped_pred = torch.flip(vflipped_pred, [-2])
    
    # 旋转90度
    rotated_images = torch.rot90(images, k=1, dims=[-2, -1])
    with torch.no_grad():
        rotated_pred = model(rotated_images)
    # 将旋转后的预测结果再旋转回来
    rotated_pred = torch.rot90(rotated_pred, k=3, dims=[-2, -1])
    
    # 平均所有预测结果
    final_pred = (orig_pred + flipped_pred + vflipped_pred + rotated_pred) / 4
    
    return final_pred

def get_class_weights(mask_dir, num_classes=4):
    class_freq = np.zeros(num_classes)
    for img_name in os.listdir(mask_dir):
        mask = cv2.imread(os.path.join(mask_dir, img_name), 0)
        for cls in range(num_classes):
            class_freq[cls] += np.sum(mask == cls)
    
    total = class_freq.sum()
    class_weights = total / (class_freq*num_classes) # 逆频率权重
    return class_weights

def validate(model_path, use_tta=False, use_rgb=True, use_hsv=False, use_lab=False, color_space_name="RGB"):
    """
    验证模型在验证集上的性能
    :param model_path: 模型路径
    :param use_tta: 是否使用测试时增强
    :param use_rgb: 是否使用RGB通道
    :param use_hsv: 是否使用HSV通道
    :param use_lab: 是否使用LAB通道
    :param color_space_name: 颜色空间名称
    """
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4  # 0:背景, 1:健康, 2:死亡, 3:白化
    BATCH_SIZE = 16
    
    # 记录验证配置
    logger, log_file = setup_logger(log_dir='logs', log_name='validation')
    logger.info("="*60)
    logger.info(f"Starting validation with model: {model_path}")
    logger.info(f"Using TTA: {use_tta}")
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"Color space: {color_space_name}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Num Classes: {NUM_CLASSES}")
    logger.info("="*60)
    
    # 加载验证数据
    _, val_loader = get_data_loaders(
        train_image_dir='dataset/train/train/images_1',
        train_mask_dir='dataset/train/train/masks',
        val_image_dir='dataset/train/val/images_1',
        val_mask_dir='dataset/train/val/masks',
        batch_size=BATCH_SIZE,
    )
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # 初始化模型
    model = MultiModalSegFormerB5(
        num_classes=NUM_CLASSES,
        use_rgb=use_rgb,
        use_hsv=use_hsv,
        use_lab=use_lab,
        pretrained=False
    ).to(DEVICE)
    
    # 加载模型权重
    logger.info(f"Loading model weights from {model_path}...")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    try:
    # 尝试作为checkpoint字典加载（包含'model_state_dict'）
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}, best mIoU: {checkpoint.get('best_miou', 'unknown'):.4f}")
    except TypeError:
        # 如果checkpoint不是字典，说明是直接的state_dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded model from direct weight file")
    except KeyError:
        # 如果'model_state_dict'不存在，说明是直接的state_dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded model from direct weight file")


    class_weights = get_class_weights('dataset/train/train/masks')
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    # 设置损失函数
    ce_loss = CrossEntropyLoss(ignore_index=255, weight=class_weights_tensor)
    dice_loss = DiceLoss()
    boundary_loss = BoundaryLoss(beta=0.1)
    cve_loss = CVELoss(num_classes=NUM_CLASSES, cv_weight=0.5, ignore_index=255)
    lov_loss = LovaszSoftmaxLoss(classes='present')
    
    # 开始验证
    logger.info("Starting validation...")
    model.eval()
    val_loss = 0.0
    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=DEVICE)
    total_correct = 0
    total_labeled = 0
    
    with torch.no_grad():
        for images, masks,_ in tqdm(val_loader, desc="Validating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # 根据是否启用TTA进行预测
            if use_tta:
                pred = tta_predict(model, images, DEVICE)
            else:
                pred = model(images)
            bias = torch.tensor(
                [-0.25, 0.0, 0.55, -0.05],
                device=pred.device
            ).view(1, 4, 1, 1)

            pred = pred + bias
            # 计算验证损失
            ce = ce_loss(pred, masks)
            dice = dice_loss(pred, masks)
            boundary = boundary_loss(pred, masks)
            cve = cve_loss(pred, masks)
            lov = lov_loss(pred, masks)
            
            batch_val_loss = (
                0.3 * ce +
                0.3 * dice +
                0.0 * boundary +
                0.6 * cve +
                0.0 * lov
            )
            
            val_loss += batch_val_loss.item()
            
            # 获取所有指标
            # 更新混淆矩阵
            hist += fast_hist(pred, masks, NUM_CLASSES)

            # 计算像素准确率
            pred_label = torch.argmax(pred, dim=1)
            valid = masks != 255
            total_correct += (pred_label[valid] == masks[valid]).sum().item()
            total_labeled += valid.sum().item()
    
    # 计算平均指标
    # 平均损失仍然按 batch 计算
    avg_val_loss = val_loss / len(val_loader)

    # 基于全局统计的像素准确率
    avg_pixel_acc = total_correct / (total_labeled + 1e-10)

    # 基于混淆矩阵的 IoU 和 mIoU
    avg_mIoU, avg_iou_per_class = compute_iou_from_hist(hist)
    acc_per_class, precision_per_class, confusion_prob = compute_detailed_metrics(hist)
    # 打印验证结果
    logger.info(f"\n{'='*60}")
    logger.info("Validation Results:")
    logger.info(f"{'='*60}")
    class_names = ["Background", "Healthy", "Dead", "Molded"]

    logger.info(f"\n{'='*60}")
    logger.info("Per-class Metrics:")
    logger.info(f"{'='*60}")

    for cls in range(NUM_CLASSES):
        logger.info(
            f"{class_names[cls]:12s}: "
            f"IoU={avg_iou_per_class[cls]:.4f} | "
            f"Acc={acc_per_class[cls]:.4f} | "
            f"Prec={precision_per_class[cls]:.4f}"
        )

    logger.info(f" {'Overall mIoU':12s}: {avg_mIoU:.4f}")
    logger.info(f"{'='*60}")

    logger.info("Confusion Probability Matrix (P(pred=j | gt=i)):")
    logger.info(f"{'='*60}")

    for i in range(NUM_CLASSES):
        probs = " ".join([f"{confusion_prob[i][j]:.3f}" for j in range(NUM_CLASSES)])
        logger.info(f"GT {class_names[i]:10s} -> {probs}")
    
    # 记录最终指标
    logger.info(f"\n{'='*60}")
    logger.info(f"Validation Summary:")
    logger.info(f" Val Loss: {avg_val_loss:.4f}")
    logger.info(f" Val Acc: {avg_pixel_acc:.4f}")
    logger.info(f" Val mIoU: {avg_mIoU:.4f}")
    logger.info(f" Log file: {log_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Validation completed!")
    logger.info(f"Best mIoU: {avg_mIoU:.4f}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*60}\n")
    
    return avg_val_loss, avg_pixel_acc, avg_mIoU, avg_iou_per_class


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate model on validation set')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model weights')
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Whether to use Test Time Augmentation')
    parser.add_argument('--color_space', type=str, default='RGB',
                        choices=['RGB', 'HSV', 'LAB', 'RGB_HSV', 'RGB_LAB', 'RGB_HSV_LAB'],
                        help='Color space to use for validation')
    
    args = parser.parse_args()
    
    # 解析颜色空间参数
    use_rgb = 'RGB' in args.color_space
    use_hsv = 'HSV' in args.color_space
    use_lab = 'LAB' in args.color_space
    
    # 运行验证
    validate(
        model_path=args.model_path,
        use_tta=args.use_tta,
        use_rgb=use_rgb,
        use_hsv=use_hsv,
        use_lab=use_lab,
        color_space_name=args.color_space
    )