import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def lovasz_grad(gt_sorted):
    """Standard Lovász gradient calculation (CVPR 2018)"""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax_flat(probs, labels):
    """Standard Lovász-Softmax for multi-class segmentation (CVPR 2018)"""
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue  # Skip empty classes
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=probs.device)
    
    return torch.mean(torch.stack(losses))

class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax Loss (CVPR 2018) - STRICT IMPLEMENTATION"""
    def __init__(self, classes='present', ignore_index=None):
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        logits: [B, C, H, W] (raw model outputs)
        labels: [B, H, W] (0-3 class indices)
        """
        # 1. Apply ignore_index mask
        if self.ignore_index is not None:
            # ignore修改
            mask = (labels != self.ignore_index)
        else:
            mask = torch.ones_like(labels, dtype=torch.bool)
        
        # 2. Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # 3. Reshape for flat processing
        B, C, H, W = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = labels.view(-1)

        # ignore修改
        mask_flat = mask.view(-1)
        probs_flat = probs_flat[mask_flat]
        labels_flat = labels_flat[mask_flat]
        
        # 4. Compute Lovász loss (excluding background class 0)
        if self.classes == 'present':
            non_bg_mask = (labels_flat != 0)
            probs_flat_bg = probs_flat[non_bg_mask]
            labels_flat_bg = labels_flat[non_bg_mask]
            loss = lovasz_softmax_flat(probs_flat_bg, labels_flat_bg)
        else:  # classes == 'all'
            loss = lovasz_softmax_flat(probs_flat, labels_flat)
        
        # 5. Add gradient stability term (per CVPR 2018)
        loss = loss + 1e-6 * torch.norm(logits, p=2)
        
        return loss
    
class BoundaryLoss(nn.Module):
    """Boundary Loss (MICCAI 2023) - STRICT IMPLEMENTATION"""
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
        
        # 1. 定义 Sobel 算子核 (3x3)
        sobel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=torch.float32
        )
        
        # 2. 不预先分配设备，而是将其作为未初始化的buffer
        # 在forward中根据输入张量的设备动态创建卷积核
        self.register_buffer('sobel_template', sobel.view(1, 1, 3, 3), persistent=False)

    def forward(self, pred, target):
        # 获取输入张量的设备
        device = pred.device
        
        # 动态创建对应设备上的卷积核
        sobel_kernel = self.sobel_template.to(device)
        
        # 创建 mask：0 且不是 255 的才是前景
        # 注意：255 既不是背景 0，也不是有效前景 1-3
        valid_mask = (target != 255).float().unsqueeze(1) 
        pred_softmax = F.softmax(pred, dim=1)
        pred_label = torch.argmax(pred_softmax, dim=1, keepdim=True).float()
        # 只有 valid 区域参与边界计算
        pred_bound = F.conv2d(pred_label * valid_mask, sobel_kernel, padding=1)
        target_bound = F.conv2d(target.unsqueeze(1).float() * valid_mask, sobel_kernel, padding=1)
    
        # 只在有效区域计算 MSE
        return self.beta * F.mse_loss(pred_bound * valid_mask, target_bound * valid_mask)
        
class DiceLoss(nn.Module):
    """Dice Loss (Standard Academic Implementation)"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_softmax = F.softmax(pred, dim=1)
        
        # ✅ 关键修复：定义 valid_mask（排除 255 作为 ignore_index）
        valid_mask = (target != 255).float().unsqueeze(1)  # 创建有效区域掩码
        
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        dice_loss = 0.0
        for cls in range(pred.size(1)):
            # ✅ 使用已定义的 valid_mask
            p = (pred_softmax[:, cls, :, :] * valid_mask.squeeze(1)).reshape(-1)
            t = (target_one_hot[:, cls, :, :] * valid_mask.squeeze(1)).reshape(-1)
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice)
        return dice_loss / pred.size(1)
    
class CVELoss(nn.Module):
    """Cross-Entropy with Variance Regularization (CVPR 2021)"""
    def __init__(self, num_classes, ce_weight=0.7, cv_weight=0.3, smooth=1e-6, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.cv_weight = cv_weight
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        # 1. CE 部分支持 ignore
        ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='mean')

        # 2. Variance 部分支持 ignore
        pred_softmax = F.softmax(pred, dim=1)
        mask = (target != self.ignore_index).float().unsqueeze(1)

        # 只计算有效像素的方差
        var_map = torch.var(pred_softmax, dim=1, unbiased=False) 
        var = (var_map * mask.squeeze(1)).sum() / (mask.sum() + 1e-6)

        return self.ce_weight * ce + self.cv_weight * var
    
class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None, weight=None, label_smoothing=0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=weight,
            label_smoothing=label_smoothing
        )
        

    def forward(self, pred, target):
        return self.ce_loss(pred, target)