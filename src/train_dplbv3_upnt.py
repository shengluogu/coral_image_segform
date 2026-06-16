# train.py 
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dplbv3plus_upernet import MultiModalUPerNet
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
def setup_logger(log_dir='logs', log_name='train_dpl_upn'):
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
    log_file = os.path.join(log_dir, f'{log_name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
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
        
def calculate_metrics(pred, target, num_classes):
    """计算像素准确率、每个类别的IoU和mIoU (学术级实现)"""
    pred = torch.argmax(pred, dim=1)
    
    # 有效像素掩码 (排除忽略类别)
    valid = (target >= 0) & (target < num_classes)
    
    # 像素准确率
    correct = (pred == target) & valid
    pixel_acc = correct.sum().float() / valid.sum().float()
    
    # mIoU (逐类别计算)
    iou_per_class = []
    for cls in range(num_classes):
        true_class = (target == cls) & valid
        pred_class = (pred == cls) & valid
        intersection = (true_class & pred_class).sum().float()
        union = (true_class | pred_class).sum().float()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(1.0, device=target.device)
        
        iou_per_class.append(iou)
    
    mIoU = torch.stack(iou_per_class).mean()
    return pixel_acc.item(), mIoU.item(), iou_per_class

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
def train(use_rgb=True, use_hsv=False, use_lab=False, color_space_name="RGB"):
    os.makedirs('weights_upernet', exist_ok=True)
    DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4  # 0:背景, 1:健康, 2:死亡, 3:白化
    BATCH_SIZE = 16
    EPOCHS = 600
    LEARNING_RATE = 0.001  # 学术级合理值
    
    # 记录训练配置
    logger.info("="*60)
    logger.info(f"Starting training with color space: {color_space_name}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"using model: UPerNet with ResNet101 backbone")
    logger.info(f"Num Classes: {NUM_CLASSES}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info("="*60)
    
    # 加载数据
    logger.info("Loading data loaders...")
    train_loader, val_loader = get_data_loaders(
        train_image_dir='dataset/train/train/images',
        train_mask_dir='dataset/train/train/masks',
        val_image_dir='dataset/train/val/images',
        val_mask_dir='dataset/train/val/masks',
        batch_size=BATCH_SIZE,
    )
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # 模型初始化
    model = MultiModalUPerNet(
        num_classes=NUM_CLASSES,
        use_rgb=use_rgb,
        use_hsv=use_hsv,
        use_lab=use_lab
    ).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算权重
    logger.info("Calculating class weights...")
    class_weights = get_class_weights('dataset/train/train/masks')
    logger.info(f"Class weights: {class_weights}")

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # 在损失函数中使用
    ce_loss = CrossEntropyLoss(ignore_index=255, weight=class_weights_tensor)
    dice_loss = DiceLoss()
    boundary_loss = BoundaryLoss(beta=0.1)
    cve_loss = CVELoss(num_classes=NUM_CLASSES, cv_weight=0.5, ignore_index=255)
    lov_loss = LovaszSoftmaxLoss(classes='present')
    
    # 专业级优化器 (AdamW + 学习率调度)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # ====== 关键修改：添加预热 + 余弦退火 ======
    warmup_epochs = max(1, int(EPOCHS * 0.10))
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        T_max=EPOCHS,
        eta_min=1e-8
    )
    

    best_miou = 0.0
    
    # === 训练循环 ===
    logger.info("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        # 使用tqdm但不显示在控制台（因为日志会记录）
        for batch_idx, (images, masks) in enumerate(train_loader):
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
            

        
        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        total_pixel_acc = 0
        total_mIoU = 0
        num_batches = 0
        total_iou_per_class = [0.0] * NUM_CLASSES
        

        with torch.no_grad():
            for images, masks in val_loader:
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
                pixel_acc, miou, iou_per_class = calculate_metrics(pred, masks, NUM_CLASSES)
                total_pixel_acc += pixel_acc
                total_mIoU += miou
                num_batches += 1
                
                # 累加每个类别的IoU
                for i in range(NUM_CLASSES):
                    total_iou_per_class[i] += iou_per_class[i].item()
        
        # 计算平均指标
        avg_val_loss = val_loss / num_batches
        avg_pixel_acc = total_pixel_acc / num_batches
        avg_mIoU = total_mIoU / num_batches
        # 计算每个类别的平均IoU
        avg_iou_per_class = [total_iou_per_class[i] / num_batches for i in range(NUM_CLASSES)]
        

        # === 打印每个类别的IoU ===
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1} Validation Results:")
        logger.info(f"{'='*30}")
        class_names = ["Background", "Healthy", "Dead", "Molded"]
        for cls in range(NUM_CLASSES):
            logger.info(f"  {class_names[cls]:12s}: IoU = {avg_iou_per_class[cls]:.4f}")
        logger.info(f"  {'Overall mIoU':12s}: {avg_mIoU:.4f}")
        logger.info(f"{'='*30}")
        
        # 记录详细指标
        logger.info(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        logger.info(f"  Train Loss:    {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:      {avg_val_loss:.4f}")
        logger.info(f"  Val Acc:       {avg_pixel_acc:.4f}")
        logger.info(f"  Val mIoU:      {avg_mIoU:.4f}")
        logger.info(f"  Best mIoU:     {best_miou:.4f}")
        logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        # 保存最佳模型
        if avg_mIoU > best_miou:
            best_miou = avg_mIoU
            model_path = f'weights_upernet/best_{color_space_name}_model.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ New best model saved at epoch {epoch+1} (mIoU: {best_miou:.4f}) -> {model_path}")
        
        # 学习率调度
        scheduler.step()
        
        # 每50个epoch保存一次检查点
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f'weights_upernet/checkpoint_{color_space_name}_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    

    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed for {color_space_name}!")
    logger.info(f"Best mIoU: {best_miou:.4f}")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"{'='*60}\n")
    
    return best_miou

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
        rgb_miou = train(use_rgb=True, use_hsv=False, use_lab=False, color_space_name="RGB")
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("Final Training Results:")
        logger.info("="*60)
        logger.info(f"RGB Model Best mIoU:    {rgb_miou:.4f}")
        logger.info(f"Total Training Time:    {total_time/3600:.2f} hours")
        logger.info(f"Log File Location:      {log_file_path}")
        logger.info("="*60)
        
        # 保存最佳模型信息到单独的结果文件
        with open('results.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("Training Results Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best mIoU (RGB): {rgb_miou:.4f}\n")
            f.write(f"Total Training Time: {total_time/3600:.2f} hours\n")
            f.write(f"Log File: {log_file_path}\n")
            f.write("="*60 + "\n")
        
        logger.info("\nTraining results saved to 'results.txt'")
        logger.info("All logs have been saved to the log file.")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise