# train.py 
'''
    # ====== 加载预训练权重 ======
    resume_model_path = '/amax/chenliangyu/work/CDNet/weights_sgfmb5_best/best_RGB_model.pth'
    
    if os.path.isfile(resume_model_path):
        logger.info(f"Loading pretrained model from: {resume_model_path}")
        state_dict = torch.load(resume_model_path, map_location=DEVICE)
        # 打印数量对比
        print(f"权重文件中的参数条目数: {len(state_dict.keys())}")
        print(f"当前代码模型的参数条目数: {len(model.state_dict().keys())}")

        # 找出那几个多出来的具体是谁（除了你日志里看到的）
        extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
        print(f"多出来的键数量: {len(extra_keys)}")
        print(f"前 3 个多出来的键: {list(extra_keys)[:3]}")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")
        logger.info("Model weights loaded successfully.")
    else:
        logger.warning(f"No checkpoint found at: {resume_model_path}")
'''
from utils.dirty_data_detector import DirtyPixelDetector   # === [新增] ===
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sgfmb5 import MultiModalSegFormerB5
#from models.deeplabv3ResNet50 import MultiModalDeepLabV3Plus
#from utils.data_loader import get_data_loaders
from utils.mosiac_loader import get_data_loaders
from utils.Loss import LovaszSoftmaxLoss, CrossEntropyLoss, DiceLoss, BoundaryLoss, CVELoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import cv2
import logging
from logging.handlers import RotatingFileHandler


# ====== 配置日志系统 ======
def setup_logger(log_dir='logs', log_name='tra_sgfmb5_test2_5'):
    """配置日志记录器，同时输出到文件和控制台"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器（带轮转，防止文件过大）
    log_file = os.path.join(log_dir, f'{log_name}_{time.strftime("%y%m%d_%H%M")}.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
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

# 全局日志对象
logger, log_file_path = setup_logger()

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """预热 + 余弦退火调度器 (标准实现)"""
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max  # 总epoch数
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热: 从0.0001增长到初始学习率
            return [
                self.eta_min + (base_lr - self.eta_min) * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火: 从初始学习率衰减到eta_min
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]
        
def calculate_metrics(pred, target, num_classes, ignore_index=255):
    """
    规范的语义分割评估指标计算：
    - Pixel Accuracy
    - 每个类别的 IoU
    - mIoU（基于整个 batch 的混淆矩阵）
    """
    pred = torch.argmax(pred, dim=1)

    # 有效像素掩码
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    # 计算像素准确率
    correct = (pred == target).sum().float()
    total = valid_mask.sum().float()
    pixel_acc = correct / (total + 1e-10)

    # 构建混淆矩阵
    hist = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()

    # 计算 IoU
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection

    iou_per_class = intersection / (union + 1e-10)

    # 仅对出现过的类别求平均
    valid_classes = union > 0
    mIoU = iou_per_class[valid_classes].mean()

    return (
        pixel_acc.item(),
        mIoU.item(),
        iou_per_class.tolist(),
        hist  # 返回混淆矩阵以便在整个验证集上累计
    )
# ====== 添加类别权重计算 ======
def get_class_weights(mask_dir, num_classes=4):
    class_freq = np.zeros(num_classes)
    for img_name in os.listdir(mask_dir):
        mask = cv2.imread(os.path.join(mask_dir, img_name), 0)
        for cls in range(num_classes):
            class_freq[cls] += np.sum(mask == cls)
    
    total = class_freq.sum()
    class_weights = total / (class_freq*num_classes) # 逆频率权重
    return class_weights

# ======================
# 3. 专业级训练函数
# ======================
def train(use_rgb=True, use_hsv=False, use_lab=False,color_space_name="RGB",
        enable_dirty_detection=True,        # === [新增] 总开关 ===
        dirty_start_epoch=90,               # === [新增] 可调参数 ===
        dirty_end_epoch=200):
    weights_dir = "weights_sgfmb5_test2_5" 
    os.makedirs(weights_dir, exist_ok=True)
    DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4  # 0:背景, 1:健康, 2:死亡, 3:白化
    BATCH_SIZE = 16
    EPOCHS = 650 
    LEARNING_RATE = 5e-5  

    
    # 记录训练配置
    logger.info("="*60)
    logger.info(f"Starting training with color space: {color_space_name}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Using model: SegFormer-B5 Multimodal")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info("="*60)
    
    # 加载数据
    logger.info("Loading data loaders...")
    train_loader, val_loader = get_data_loaders(
        train_image_dir='dataset/tra_pri/tra/image',
        train_mask_dir='dataset/tra_pri/tra/label',
        val_image_dir='dataset/tra_pri/val/image',
        val_mask_dir='dataset/tra_pri/val/label',
        batch_size=BATCH_SIZE,
    )
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    # === [新增] 初始化脏数据检测器 ===
    if enable_dirty_detection:
        dirty_detector = DirtyPixelDetector(
            save_dir=f"{weights_dir}/dirty_data_analysis",
            num_classes=NUM_CLASSES,
            ignore_index=255,
            start_epoch=dirty_start_epoch,
            end_epoch=dirty_end_epoch,
            ema_alpha=0.1,
            topk_percent_per_class=5.0,
            class_names=["Background", "Healthy", "Dead", "Molded"],
        )
        logger.info(f"✓ DirtyPixelDetector ENABLED: window=[{dirty_start_epoch}, {dirty_end_epoch}]")
    else:
        dirty_detector = None
        logger.info("✗ DirtyPixelDetector DISABLED")

    # 模型初始化
    model = MultiModalSegFormerB5(
        num_classes=NUM_CLASSES,
        use_rgb=use_rgb,
        use_hsv=use_hsv,
        use_lab=use_lab,
        pretrained=True  # 加载预训练权重
    ).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算权重
    logger.info("Calculating class weights...")
    class_weights = get_class_weights('dataset/tra_pri/tra/label')
    logger.info(f"Class weights: {class_weights}")

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # 在损失函数中使用
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # 在损失函数中使用
    ce_loss = CrossEntropyLoss(ignore_index=255, weight=class_weights_tensor )
    dice_loss = DiceLoss()
    boundary_loss = BoundaryLoss(beta=0.1)
    cve_loss = CVELoss(num_classes=NUM_CLASSES, cv_weight=0.5, ignore_index=255)
    lov_loss = LovaszSoftmaxLoss(classes='present')
    
    # 专业级优化器 (AdamW + 学习率调度)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-2
    )
    
    # ====== 关键修改：添加预热 + 余弦退火 ======
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=80,
        T_max=EPOCHS,
        eta_min=2e-7
    )

    best_miou1 = 0.0
    best_epoch1 = -1
    best_miou2 = 0.0
    best_epoch2 = -1

    # === 训练循环 ===
    logger.info("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        # 使用tqdm但不显示在控制台（因为日志会记录）
        for batch_idx, (images, masks, filenames) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            pred = model(images)
            
            ce = ce_loss(pred, masks)
            dice = dice_loss(pred, masks)
            boundary = boundary_loss(pred, masks)
            cve = cve_loss(pred, masks)
            lov = lov_loss(pred, masks)
            
            batch_loss = (
                    0.3 * ce +
                    0.3 * dice +
                    0.0 * boundary +
                    0.6 * cve +
                    0.0 * lov
                )
            # 优化步骤
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            # === [新增] 更新脏数据检测器 ===
            if dirty_detector is not None and dirty_detector.is_in_window(epoch + 1):
                dirty_detector.update(pred, masks, filenames)
        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        total_correct = 0.0
        total_pixels = 0.0
        confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES))
        
        
        with torch.no_grad():
            for images, masks,_ in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                pred = model(images)
                
                # 验证阶段使用相同损失组合
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
                pixel_acc, miou, iou_per_class, hist = calculate_metrics(
                    pred, masks, NUM_CLASSES
                )

                confusion_matrix += hist.cpu()

                # 像素准确率累计
                total_correct += (torch.diag(hist)).sum().item()
                total_pixels += hist.sum().item()
                
        
        # 计算平均指标
        avg_val_loss = val_loss / len(val_loader)

        # Pixel Accuracy
        avg_pixel_acc = total_correct / (total_pixels + 1e-10)

        # IoU 计算
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
        iou_per_class = intersection / (union + 1e-10)

        # 仅对出现过的类别求平均
        valid_classes = union > 0
        avg_mIoU = iou_per_class[valid_classes].mean().item()
        avg_iou_per_class = iou_per_class.tolist()
        
        
        # === 打印每个类别的IoU ===
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1} Validation Results:")
        logger.info(f"{'='*30}")
        class_names = ["Background", "Healthy", "Dead", "Molded"]
        for cls in range(NUM_CLASSES):
            logger.info(f"  {class_names[cls]:12s}: IoU = {avg_iou_per_class[cls]:.4f}")
        #logger.info(f"  {'Overall mIoU':12s}: {avg_mIoU:.4f}")
        logger.info(f"{'='*30}")
        # 保存最佳模型
        best1_path = f'{weights_dir}/best1_{color_space_name}_model.pth'
        best2_path = f'{weights_dir}/best2_{color_space_name}_model.pth'
        if avg_mIoU > best_miou1:

            # 先更新 second best
            best_miou2 = best_miou1
            best_epoch2 = best_epoch1
            # 保存旧 best1 的参数到 best2
            if os.path.exists(best1_path):
                best1_state = torch.load(best1_path)
                torch.save(best1_state, best2_path)

            # 更新 best1
            best_miou1 = avg_mIoU
            best_epoch1 = epoch + 1
            # 保存新的 best1
            torch.save(model.state_dict(), best1_path)

            logger.info(
                f"✓ New best1 saved at epoch {epoch+1} "
                f"(mIoU: {best_miou1:.4f})"
            )

        elif avg_mIoU > best_miou2:

            best_miou2 = avg_mIoU
            best_epoch2 = epoch + 1
            # 保存新的 best2
            torch.save(model.state_dict(), best2_path)

            logger.info(
                f"✓ New best2 saved at epoch {epoch+1} "
                f"(mIoU: {best_miou2:.4f})"
            )

        # 记录详细指标
        logger.info(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        logger.info(f"  Train Loss:    {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:      {avg_val_loss:.4f}")
        logger.info(f"  Val Acc:       {avg_pixel_acc:.4f}")
        logger.info(f"  Val mIoU:      {avg_mIoU:.4f}")
        logger.info(f"  Best mIoU1:     {best_miou1:.4f} (Epoch {best_epoch1})")
        logger.info(f"  Best mIoU2:     {best_miou2:.4f} (Epoch {best_epoch2})")
        logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.8f}")

        # === [新增] 到达 end_epoch 时导出脏数据分析 ===
        if dirty_detector is not None and dirty_detector.should_export(epoch + 1):
            logger.info("="*60)
            logger.info(f"[DirtyDetector] Exporting at epoch {epoch+1}...")
            dirty_detector.export(logger=logger)
            logger.info("="*60)
        
        # 学习率调度
        scheduler.step()

        # 每80个epoch保存一次检查点
        if (epoch + 1) % 80 == 0:
            checkpoint_path = f'{weights_dir}/checkpoint_{color_space_name}_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou1,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed for {color_space_name}!")
    logger.info(f"Best mIoU: {best_miou1:.4f} (Epoch {best_epoch1})")
    logger.info(f"Second Best mIoU: {best_miou2:.4f} (Epoch {best_epoch2})")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"{'='*60}\n")

    return best_miou1, best_miou2

# ======================
# 5. 执行训练
# ======================
if __name__ == "__main__":
    logger.info("="*60)
    logger.info(" Starting Professional Lovász-Softmax DeepLabV3+ Training")
    logger.info("="*60)
    logger.info(f"Log file will be saved to: {log_file_path}")
    logger.info("="*60)
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 分别训练三种颜色空间模型
        logger.info("\nStarting RGB model training...")
        rgb_miou1, rgb_miou2 = train(
            use_rgb=True, use_hsv=False, use_lab=False, 
            color_space_name="RGB",
            enable_dirty_detection=True,        # ← 第一次开启
            dirty_start_epoch=90,
            dirty_end_epoch=200,
        )
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("Final Training Results:")
        logger.info("="*60)
        logger.info(f"RGB Model Best mIoU1:    {rgb_miou1:.4f} (Epoch {rgb_epoch1})")
        logger.info(f"RGB Model Best mIoU2:    {rgb_miou2:.4f} (Epoch {rgb_epoch2})")
        logger.info(f"Total Training Time:    {total_time/3600:.2f} hours")
        logger.info(f"Log File Location:      {log_file_path}")
        logger.info("="*60)
        
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise