import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import random
import torch
import numpy as np
from typing import List, Tuple
np.int = int

def cutmix_collate_fn(batch, alpha=1.0, apply_cutmix_prob=0.5):
    """
    自定义 collate_fn，对 batch 应用 CutMix。
    
    Args:
        batch: list of (image, mask) tuples
        alpha: Beta 分布参数
        apply_cutmix_prob: 应用 CutMix 的概率
    
    Returns:
        mixed_images: [B, C, H, W]
        mixed_masks: [B, H, W] (long)
    """
    if np.random.rand() > apply_cutmix_prob:
        # 不应用 CutMix，正常堆叠
        images, masks,_ = zip(*batch)
        return torch.stack(images, dim=0), torch.stack(masks, dim=0), [f for _, _, f in batch]

    B = len(batch)
    images, masks,_ = zip(*batch)
    images = torch.stack(images, dim=0).float()  # [B, C, H, W]
    masks = torch.stack(masks, dim=0).float()    # [B, H, W]

    # 生成 λ ~ Beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha)
    
    # 随机打乱索引
    index = torch.randperm(B)
    
    # 创建掩码
    height, width = images.shape[2], images.shape[3]
    cut_ratio = np.sqrt(1. - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    
    # 随机位置
    y = np.random.randint(0, height - cut_h)
    x = np.random.randint(0, width - cut_w)
    
    # CutMix 操作
    images[:, :, y:y+cut_h, x:x+cut_w] = images[index, :, y:y+cut_h, x:x+cut_w]
    
    # 计算混合标签
    masks[:, y:y+cut_h, x:x+cut_w] = masks[index, y:y+cut_h, x:x+cut_w]
    
    return images, masks.long(), [f for _, _, f in batch]  # 返回文件名列表

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

import cv2
import numpy as np
import random

def mosaic_augmentation(images, masks, size=256):
    yc, xc = [int(random.uniform(0.3, 0.7) * size) for _ in range(2)]
    mosaic_img = np.full((size, size, 3), 114, dtype=np.uint8)
    mosaic_mask = np.full((size, size),255, dtype=np.uint8)

    for i, (img_path, mask_path) in enumerate(zip(images, masks)):
        img = load_image(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]

        if i == 0:  # 左上
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
        else:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, size), min(size, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        mosaic_mask[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]

    return mosaic_img, mosaic_mask

class CoralDataset(Dataset):
    def __init__(self, image_dir, mask_dir, 
                 spatial_transform=None,   # 空间变换（同时作用于 image 和 mask）
                 color_transform=None,     # 颜色变换（仅 image）
                 common_transform=None,
                 use_mosaic=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.common_transform = common_transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.use_mosaic = use_mosaic
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        if self.use_mosaic and random.random() < 0.7:  # 根据一定概率应用 Mosaic
            # 选择四张图片进行 Mosaic 增强
            indices = [idx] + [random.randint(0, len(self.images)-1) for _ in range(3)]
            image_paths = [os.path.join(self.image_dir, self.images[i]) for i in indices]
            mask_paths = [os.path.join(self.mask_dir, self.images[i]) for i in indices]
            
            image, mask = mosaic_augmentation(image_paths, mask_paths, size=256)
        else:
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name)
            
            image = load_image(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 应用其他增强...
        if self.spatial_transform:
            augmented = self.spatial_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.color_transform:
            image = self.color_transform(image=image)['image']

        if self.common_transform:
            image = self.common_transform(image=image)['image']

        mask = torch.from_numpy(mask).long()

        return image, mask,img_name
    
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, 
    ElasticTransform, GridDistortion, Perspective,
    RandomResizedCrop, ShiftScaleRotate, CoarseDropout,
    RandomGridShuffle, Resize, CLAHE, RandomBrightnessContrast,
    OneOf, GaussNoise, Normalize, HueSaturationValue, OpticalDistortion,
    GridDropout, ColorJitter, RandomGamma, RandomToneCurve, Solarize,
    Posterize, ChannelShuffle, RGBShift, ISONoise, MultiplicativeNoise,
    GaussianBlur, UnsharpMask, Sharpen
)
from albumentations.pytorch import ToTensorV2

def get_data_loaders(train_image_dir, train_mask_dir, 
                     val_image_dir, val_mask_dir, 
                     batch_size=4,use_mixup=True,          # 新增参数
                     mixup_alpha=1.0):
    
    # ========== 训练集增强（增强版）==========
    train_spatial = Compose([
        # 基础尺寸
        Resize(height=256, width=256),
        
        # 尺度与视角变化
        RandomResizedCrop(height=256, width=256, scale=(0.6, 1.0), p=0.5),
        # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, 
        #                 p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        # Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
        
        # 形变类
        ElasticTransform(alpha=50, sigma=50, alpha_affine=50, p=0.3),
        # GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        # OpticalDistortion(distort_limit=0.5, shift_limit=0.1, p=0.3),
        
        # 遮挡类（新增）
        CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                     min_holes=2, min_height=16, min_width=16, 
                     fill_value=0, p=0.3),
        # GridDropout(ratio=0.5, unit_size_min=10, unit_size_max=30, 
        #            holes_number_x=5, holes_number_y=5, 
        #            random_offset=True, p=0.2),
        
        # 翻转
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        
        # 激进增强
        # RandomGridShuffle(grid=(3, 3), p=0.1),
    ], additional_targets={'mask': 'mask'})

    # 颜色变换（增强版）
    train_color = Compose([
        # 基础颜色增强
        CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        
        OneOf([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
            # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.5),
        
        # # 高级颜色增强（新增）
        # OneOf([
        #     RandomGamma(gamma_limit=(80, 120), p=1.0),
        #     RandomToneCurve(scale=0.1, p=1.0),
        #     Solarize(threshold=128, p=1.0),
        #     Posterize(num_bits=(4, 6), p=1.0),
        # ], p=0.3),
        
         #通道操作（新增）
        OneOf([
            ChannelShuffle(p=0.05),
            RGBShift(r_shift_limit=(-20, 5),g_shift_limit=(-10, 10),b_shift_limit=(-5, 15), p=1.0),
        ], p=0.4),

        
        # # 噪声增强（增强版）
        # OneOf([
        #     GaussNoise(var_limit=(10, 50), p=1.0),
        #     ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        #     MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        # ], p=0.4),
        
        # 模糊增强（新增）
        OneOf([
            GaussianBlur(blur_limit=(3, 7), p=1.0),
            # MotionBlur(blur_limit=7, p=1.0),
            # MedianBlur(blur_limit=5, p=1.0),
            # AdvancedBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        
        # # 天气效果（新增，低概率）
        # OneOf([
        #     RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1.0),
        #     RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, 
        #               drop_width=1, drop_color=(200, 200, 200), p=1.0),
        #     RandomShadow(num_shadows_lower=1, num_shadows_upper=3, 
        #                 shadow_dimension=5, p=1.0),
        # ], p=0.15),
        
        # 锐化增强（新增）
        # OneOf([
        #     UnsharpMask(blur_limit=(3, 7), sigma_limit=0.5, p=1.0),
        #     Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        # ], p=0.2),
        
        # # 压缩失真（新增）
        # ImageCompression(quality_lower=50, quality_upper=95, p=0.2),
    ])

    # 通用变换（仅 image）
    train_common = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


    # ========== 验证集（无增强） ==========
    val_common = Compose([
        Resize(height=256, width=256),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_dataset = CoralDataset(
        train_image_dir, train_mask_dir,
        spatial_transform=train_spatial,
        color_transform=train_color,
        common_transform=train_common
    )

    # 验证集不需要 Mixup
    val_dataset = CoralDataset(
        val_image_dir, val_mask_dir,
        spatial_transform=None,
        color_transform=None,
        common_transform=val_common,
        use_mosaic=False
    )

    # 训练 DataLoader：使用自定义 collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=lambda batch: cutmix_collate_fn(batch, alpha=mixup_alpha, apply_cutmix_prob=0.5) if use_mixup else torch.utils.data.default_collate(batch)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True
        # 验证集不用 Mixup，用默认 collate
    )
    
    return train_loader, val_loader