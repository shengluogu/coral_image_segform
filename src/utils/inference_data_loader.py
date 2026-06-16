# utils/inference_data_loader.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

# utils/inference_data_loader.py
class InferenceCoralDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        self.image_names = [os.path.basename(p) for p in self.image_paths]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 1. 用cv2读取图像（默认BGR格式）
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        
        # 2. ✅ 关键修复：将BGR转为RGB（与训练时一致）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. 转换为numpy数组并归一化
        img = np.array(img, dtype=np.float32)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        # 4. 转换为PyTorch张量 (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img, self.image_names[idx]