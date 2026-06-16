import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class CoralDataset(Dataset):
    def __init__(self, image_dir, mask_dir, 
                 spatial_transform=None,   # 空间变换（同时作用于 image 和 mask）
                 color_transform=None,     # 颜色变换（仅 image）
                 common_transform=None):   # 归一化、ToTensor（仅 image）
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.common_transform = common_transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)  # 单通道，H x W

        # 1. 空间变换（同时作用 image 和 mask）
        if self.spatial_transform:
            augmented = self.spatial_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 2. 颜色变换（仅 image）
        if self.color_transform:
            image = self.color_transform(image=image)['image']

        # 3. 通用变换（仅 image 的归一化和 ToTensor）
        if self.common_transform:
            image = self.common_transform(image=image)['image']

        # mask 转换为 LongTensor（CrossEntropyLoss 要求）
        mask = torch.from_numpy(mask).long()  # [H, W]

        return image, mask
    
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate,
    RandomBrightnessContrast, HueSaturationValue, Normalize, RGBShift
    ,CLAHE, CoarseDropout, OneOf, GaussNoise
)
from albumentations.pytorch import ToTensorV2

def get_data_loaders(train_image_dir, train_mask_dir, 
                     val_image_dir, val_mask_dir, 
                     batch_size=4):
    # ========== 训练集增强 ==========
    # 空间变换（同时作用于 image 和 mask）
    train_spatial = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        # CoarseDropout(max_holes=4, max_height=8, max_width=8, 
        #           min_holes=2, min_height=4, min_width=4, 
        #           fill_value=0, p=0.3),
    ], additional_targets={'mask': 'mask'})  # 明确 mask 作为第二个目标

    # 颜色变换（仅 image）
    train_color = Compose([
        CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        # RGBShift(r_shift_limit=(-30, 0), g_shift_limit=(-5, 15), b_shift_limit=(0, 15), p=0.3),
        OneOf([
            # IAAAdditiveGaussianNoise(scale=(0, 0.02*255), p=0.5),
            GaussNoise(var_limit=(10, 50), p=0.5),
        ], p=0.3),
    ])

    # 通用变换（仅 image）
    train_common = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_dataset = CoralDataset(
        train_image_dir, train_mask_dir,
        spatial_transform=train_spatial,
        color_transform=train_color,
        common_transform=train_common
    )

    # ========== 验证集（无增强） ==========
    val_common = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_dataset = CoralDataset(
        val_image_dir, val_mask_dir,
        spatial_transform=None,
        color_transform=None,
        common_transform=val_common
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )
    return train_loader, val_loader