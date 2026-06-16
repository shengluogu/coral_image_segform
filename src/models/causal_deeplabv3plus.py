# models/causal_deeplabv3plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from models.underwater_enhancer import enhance_underwater_image
import cv2
import numpy as np

def resnet_atrous_adaption(model, output_stride):
    """修改ResNet以支持空洞卷积"""
    if output_stride == 16:
        model.layer4[0].conv2.stride = (2, 2)
        model.layer4[0].downsample[0].stride = (2, 2)
    elif output_stride == 8:
        model.layer3[0].conv2.stride = (2, 2)
        model.layer3[0].downsample[0].stride = (2, 2)
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        
        # 应用空洞卷积
        for idx in range(1, len(model.layer3)):
            model.layer3[idx].conv2.dilation = (2, 2)
            model.layer3[idx].conv2.padding = (2, 2)
        for idx in range(1, len(model.layer4)):
            model.layer4[idx].conv2.dilation = (4, 4)
            model.layer4[idx].conv2.padding = (4, 4)
    return model

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, output_stride):
        super().__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

class Decoder(nn.Module):
    """Decoder module"""
    def __init__(self, high_level_ch, low_level_ch, num_classes):
        super().__init__()
        # 低层特征通道数减少
        self.conv1 = nn.Conv2d(low_level_ch, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        
        # 最终分类
        self.last_conv = nn.Sequential(
            nn.Conv2d(high_level_ch + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        
    def forward(self, high_level_feat, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        high_level_feat = F.interpolate(high_level_feat, size=low_level_feat.size()[2:], 
                                      mode='bilinear', align_corners=True)
        x = torch.cat((high_level_feat, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

def rgb_to_hsv_batch(rgb_tensor):
    rgb_uint8 = (rgb_tensor * 255).clamp(0, 255).byte()
    hsv_list = []
    for i in range(rgb_uint8.shape[0]):
        rgb_np = rgb_uint8[i].cpu().numpy().transpose(1, 2, 0)
        hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        h = hsv_np[:, :, 0] / 179.0
        s = hsv_np[:, :, 1] / 255.0
        v = hsv_np[:, :, 2] / 255.0
        hsv_correct = np.stack([h, s, v], axis=2)    
        hsv_tensor = torch.from_numpy(hsv_correct.transpose(2, 0, 1)).float()
        hsv_list.append(hsv_tensor)
    return torch.stack(hsv_list).to(rgb_tensor.device)

def rgb_to_lab_batch(rgb_tensor):
    lab_list = []
    rgb_uint8 = (rgb_tensor * 255).clamp(0, 255).byte()
    for i in range(rgb_uint8.shape[0]):
        rgb_np = rgb_uint8[i].cpu().numpy().transpose(1, 2, 0)
        lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        # 可选：使用 256 更精确（但 255 也可接受）
        lab_normalized = (lab_np + [0, 128, 128]) / [100, 256, 256]
        lab_tensor = torch.from_numpy(lab_normalized.transpose(2, 0, 1)).float()
        lab_list.append(lab_tensor)
    return torch.stack(lab_list).to(rgb_tensor.device)

class MultiModalDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=4, backbone='resnet101', output_stride=16,
                 use_rgb=True, use_hsv=False, use_lab=False):
        """
        Args:
            use_rgb: 是否使用RGB颜色空间
            use_hsv: 是否使用HSV颜色空间
            use_lab: 是否使用LAB颜色空间
        """
        super().__init__()
        
        # 仅初始化需要的颜色空间分支
        self.use_rgb = use_rgb
        self.use_hsv = use_hsv
        self.use_lab = use_lab
        
        # 初始化backbone（仅当需要时）
        if use_rgb:
            self.rgb_backbone = resnet101(pretrained=False, progress=False)
            self._load_resnet_weights(self.rgb_backbone)
        if use_hsv:
            self.hsv_backbone = resnet101(pretrained=False, progress=False)
            self._load_resnet_weights(self.hsv_backbone)
        if use_lab:
            self.lab_backbone = resnet101(pretrained=False, progress=False)
            self._load_resnet_weights(self.lab_backbone)
        
        # 适配空洞卷积
        if use_rgb:
            self.rgb_backbone = resnet_atrous_adaption(self.rgb_backbone, output_stride)
        if use_hsv:
            self.hsv_backbone = resnet_atrous_adaption(self.hsv_backbone, output_stride)
        if use_lab:
            self.lab_backbone = resnet_atrous_adaption(self.lab_backbone, output_stride)
        
        # 计算动态输入通道数
        in_channels = 0
        low_level_ch = 0
        if use_rgb:
            in_channels += 2048
            low_level_ch += 256
        if use_hsv:
            in_channels += 2048
            low_level_ch += 256
        if use_lab:
            in_channels += 2048
            low_level_ch += 256
        
        # ASPP和解码器使用动态输入通道
        self.aspp = ASPP(in_channels=in_channels, out_channels=256, output_stride=output_stride)
        self.decoder = Decoder(high_level_ch=256, low_level_ch=low_level_ch, num_classes=num_classes)

    def _load_resnet_weights(self, model):
        """安全加载ResNet权重（避免文件名错误）"""
        try:
            state_dict = torch.load('./resnet101-cd907fc2.pth', map_location='cpu')
            model.load_state_dict(state_dict)
            print("✅ Weight loaded for backbone")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}, using random initialization")

    def forward(self, x):
        input_size = x.shape[2:]
        features = []  # 存储所有启用的颜色空间特征

        if self.use_rgb:
            # 1. 将输入x从归一化范围(0-1)转换回0-255 (BGR, uint8)
            x_rgb = x.cpu().detach().numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)
            x_rgb = (x_rgb * 255).astype(np.uint8)
            
            # 2. 转换为BGR（cv2使用BGR格式）
            x_rgb_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_rgb]
            
            # 3. 对每个图像应用CLAHE增强
            enhanced_list = []
            for img in x_rgb_bgr:
                enhanced_list.append(enhance_underwater_image(img))
            
            # 4. 转换回RGB并归一化
            x_rgb_enhanced = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in enhanced_list]
            x_rgb_enhanced = np.stack(x_rgb_enhanced, axis=0).astype(np.float32) / 255.0
            
            # 5. 转换为torch tensor并移回GPU
            x_rgb_enhanced = torch.from_numpy(x_rgb_enhanced).permute(0, 3, 1, 2).float().to(x.device)
            x_rgb = x_rgb_enhanced  # 用增强后的图像作为RGB分支输入
        else:
            x_rgb = x  # 如果不使用RGB分支，保持原始输入
        
        # RGB分支（如果启用）
        if self.use_rgb:
            rgb_feat = self.extract_features(self.rgb_backbone, x)
            features.append(rgb_feat)
        
        # HSV分支（如果启用）
        if self.use_hsv:
            hsv = rgb_to_hsv_batch(x)
            hsv_feat = self.extract_features(self.hsv_backbone, hsv)
            features.append(hsv_feat)
        
        # LAB分支（如果启用）
        if self.use_lab:
            lab = rgb_to_lab_batch(x)
            lab_feat = self.extract_features(self.lab_backbone, lab)
            features.append(lab_feat)
        
        # 特征融合（动态拼接）
        if not features:
            raise ValueError("At least one color space must be enabled")
        
        high_level_feat = torch.cat([f['high'] for f in features], dim=1)
        low_level_feat = torch.cat([f['low'] for f in features], dim=1)
        
        # ASPP + 解码
        high_level_feat = self.aspp(high_level_feat)
        seg_logits = self.decoder(high_level_feat, low_level_feat)
        
        return F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=True)

    def extract_features(self, backbone, x):
        """提取高低层特征（通用方法）"""
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        
        low_level_feat = backbone.layer1(x)  # 低层特征
        x = backbone.layer2(low_level_feat)
        x = backbone.layer3(x)
        high_level_feat = backbone.layer4(x)  # 高层特征
        
        return {'low': low_level_feat, 'high': high_level_feat}