import os
import shutil
import random
import numpy as np
from PIL import Image

def create_dataset_splits(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    创建训练/验证/测试集划分
    :param data_dir: 原始数据集根目录（包含images和masks子目录）
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    """
    # 确保比例总和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    
    # 创建输出目录
    os.makedirs(os.path.join(data_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'masks'), exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(os.path.join(data_dir, 'image')) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 随机打乱
    random.seed(42)
    random.shuffle(image_files)
    
    # 计算划分大小
    total = len(image_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # 划分数据集
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size+val_size]
    test_files = image_files[train_size+val_size:]
    
    # 复制文件到相应目录
    def copy_files(file_list, src_dir, dest_dir):
        for file in file_list:
            src_img = os.path.join(src_dir, 'image', file)
            src_mask = os.path.join(src_dir, 'label', file.replace('.jpg', '.png').replace('.jpeg', '.png'))
            
            # 确保掩码文件存在
            if not os.path.exists(src_mask):
                # 尝试其他掩码扩展名
                mask_ext = '.png' if file.endswith('.jpg') else '.jpg'
                src_mask = os.path.join(src_dir, 'masks', file.replace('.jpg', mask_ext).replace('.jpeg', mask_ext))
            
            if os.path.exists(src_img) and os.path.exists(src_mask):
                shutil.copy(src_img, os.path.join(dest_dir, 'images', file))
                shutil.copy(src_mask, os.path.join(dest_dir, 'masks', file.replace('.jpg', '.png').replace('.jpeg', '.png')))
    
    # 复制到训练集
    copy_files(train_files, data_dir, os.path.join(data_dir, 'train'))
    # 复制到验证集
    copy_files(val_files, data_dir, os.path.join(data_dir, 'val'))
    # 复制到测试集
    copy_files(test_files, data_dir, os.path.join(data_dir, 'test'))
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")
    print(f"  Test:  {len(test_files)} images")

# 使用示例
if __name__ == "__main__":
    # 假设原始数据集在 data/coral
    create_dataset_splits('./dataset/train', train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)