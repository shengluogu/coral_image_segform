'''
python validate_hv.py --model_path weights/checkpoint_RGB_epoch50.pth --use_tta
python validate_hv.py --model_path weights_sgfmb5/best_RGB_model.pth --use_tta
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
    测试时增强预测（TTA - 少数服从多数 Majority Voting）
    对图像进行多种变换后预测，并通过像素级多数投票融合结果。
    返回与原模型一致的 logits 形式，保证后续代码兼容。
    """
    model.eval()
    NUM_CLASSES = model.num_classes if hasattr(model, 'num_classes') else 4

    with torch.no_grad():
        # 1. 原始预测
        pred_orig = model(images)
        label_orig = torch.argmax(pred_orig, dim=1)

        # 2. 水平翻转
        images_hflip = torch.flip(images, [-1])
        pred_hflip = model(images_hflip)
        pred_hflip = torch.flip(pred_hflip, [-1])
        label_hflip = torch.argmax(pred_hflip, dim=1)

        # 3. 垂直翻转
        images_vflip = torch.flip(images, [-2])
        pred_vflip = model(images_vflip)
        pred_vflip = torch.flip(pred_vflip, [-2])
        label_vflip = torch.argmax(pred_vflip, dim=1)

        # 4. 旋转90°
        images_rot = torch.rot90(images, k=1, dims=[-2, -1])
        pred_rot = model(images_rot)
        pred_rot = torch.rot90(pred_rot, k=3, dims=[-2, -1])
        label_rot = torch.argmax(pred_rot, dim=1)

    # ============================
    # Majority Voting（少数服从多数）
    # ============================
    # shape: [N, H, W, num_transforms]
    stacked_labels = torch.stack(
        [label_orig, label_hflip, label_vflip, label_rot], dim=-1
    )

    # 使用 one-hot 统计每个类别的票数
    votes = torch.zeros(
        stacked_labels.size(0),
        NUM_CLASSES,
        stacked_labels.size(1),
        stacked_labels.size(2),
        device=device
    )

    for cls in range(NUM_CLASSES):
        votes[:, cls] = (stacked_labels == cls).sum(dim=-1)

    # 每个像素选择票数最多的类别
    final_labels = torch.argmax(votes, dim=1)

    # ============================
    # 转换为 logits 形式以保持兼容
    # ============================
    # 使用 one-hot 作为伪 logits，保证 fast_hist 和 argmax 正常工作
    final_pred = F.one_hot(final_labels, num_classes=NUM_CLASSES) \
                   .permute(0, 3, 1, 2).float()

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
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4  # 0:背景, 1:健康, 2:死亡, 3:白化
    BATCH_SIZE = 32
    
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
        train_image_dir='dataset/train/train/images',
        train_mask_dir='dataset/train/train/masks',
        val_image_dir='dataset/train/val/images',
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
        for images, masks in tqdm(val_loader, desc="Validating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # 根据是否启用TTA进行预测
            if use_tta:
                pred = tta_predict(model, images, DEVICE)
            else:
                pred = model(images)
            
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
    # 打印验证结果
    logger.info(f"\n{'='*60}")
    logger.info("Validation Results:")
    logger.info(f"{'='*60}")
    class_names = ["Background", "Healthy", "Dead", "Molded"]
    for cls in range(NUM_CLASSES):
        logger.info(f" {class_names[cls]:12s}: IoU = {avg_iou_per_class[cls]:.4f}")
    logger.info(f" {'Overall mIoU':12s}: {avg_mIoU:.4f}")
    logger.info(f"{'='*60}")
    
    # 记录最终指标
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