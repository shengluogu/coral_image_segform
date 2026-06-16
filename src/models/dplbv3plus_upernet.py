# models/causal_upernet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from models.underwater_enhancer import enhance_underwater_image
import cv2
import numpy as np

def resnet_atrous_adaption(model, output_stride):
    """修改ResNet以支持空洞卷积 (UPerNet通常不强制要求空洞卷积，但保留以维持与原代码兼容性)"""
    if output_stride == 16:
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        for m in model.layer4.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)
    elif output_stride == 8:
        model.layer3[0].conv2.stride = (2, 2)
        model.layer3[0].downsample[0].stride = (2, 2)
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        
        for idx in range(1, len(model.layer3)):
            model.layer3[idx].conv2.dilation = (2, 2)
            model.layer3[idx].conv2.padding = (2, 2)
        for idx in range(1, len(model.layer4)):
            model.layer4[idx].conv2.dilation = (4, 4)
            model.layer4[idx].conv2.padding = (4, 4)
    return model

class PPM(nn.ModuleList):
    """Pyramid Pooling Module (PPM) - UPerNet的核心组件"""
    def __init__(self, pool_scales, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        for scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out, size=x.size()[2:], mode='bilinear', align_corners=True)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead(nn.Module):
    """UPerNet 解码头"""
    def __init__(self, in_channels_list, fpn_channels=512, num_classes=4):
        super().__init__()
        # in_channels_list 对应 [Layer1, Layer2, Layer3, Layer4] 的拼接后通道数
        
        # PPM 模块
        self.ppm = PPM(pool_scales=(1, 2, 3, 6), 
                       in_channels=in_channels_list[3], 
                       out_channels=fpn_channels // 4)
        
        # PPM 输出融合卷积
        self.ppm_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[3] + fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

        # FPN 侧边连接卷积 (1x1)
        self.fpn_in = nn.ModuleList()
        for in_channels in in_channels_list[:-1]:  # Layer 1, 2, 3
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, fpn_channels, 1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # FPN 融合后的输出卷积
        self.fpn_out = nn.ModuleList()
        for i in range(len(in_channels_list) - 1):
            self.fpn_out.append(
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 最终分类头
        self.conv_last = nn.Sequential(
            nn.Conv2d(fpn_channels * len(in_channels_list), fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(fpn_channels, num_classes, 1)
        )

    def forward(self, inputs):
        # inputs 为 [feat1, feat2, feat3, feat4]
        feat1, feat2, feat3, feat4 = inputs

        # 1. PPM path (on feat4)
        ppm_outs = [feat4] + self.ppm(feat4)
        f_ppm = self.ppm_conv(torch.cat(ppm_outs, dim=1))

        # 2. FPN path (Top-down)
        fpn_features = [f_ppm]
        for i in range(len(self.fpn_in) - 1, -1, -1): # 从 Layer 3 到 Layer 1
            lateral = self.fpn_in[i](inputs[i])
            # 上采样并相加
            top_down = F.interpolate(fpn_features[-1], size=lateral.size()[2:], 
                                     mode='bilinear', align_corners=True)
            f = lateral + top_down
            f = self.fpn_out[i](f)
            fpn_features.append(f)

        # 此时 fpn_features 顺序为 [f4, f3, f2, f1]
        fpn_features.reverse() # 统一为 [f1, f2, f3, f4]

        # 3. Concatenate all levels
        output_size = fpn_features[0].size()[2:]
        combined_features = [fpn_features[0]]
        for i in range(1, len(fpn_features)):
            combined_features.append(
                F.interpolate(fpn_features[i], size=output_size, mode='bilinear', align_corners=True)
            )
        
        x = torch.cat(combined_features, dim=1)
        x = self.conv_last(x)
        return x

class MultiModalUPerNet(nn.Module):
    def __init__(self, num_classes=4, backbone='resnet101', output_stride=16,
                 use_rgb=True, use_hsv=False, use_lab=False):
        super().__init__()
        
        self.use_rgb = use_rgb
        self.use_hsv = use_hsv
        self.use_lab = use_lab
        
        # 初始化模态分支
        def get_backbone():
            model = resnet101(pretrained=False)
            self._load_resnet_weights(model)
            return resnet_atrous_adaption(model, output_stride)

        if use_rgb: self.rgb_backbone = get_backbone()
        if use_hsv: self.hsv_backbone = get_backbone()
        if use_lab: self.lab_backbone = get_backbone()

        # 计算每一层的总通道数
        num_m = sum([use_rgb, use_hsv, use_lab])
        in_channels_list = [256 * num_m, 512 * num_m, 1024 * num_m, 2048 * num_m]
        
        # 使用 UPerHead 替换原有的 ASPP + Decoder
        self.decode_head = UPerHead(in_channels_list=in_channels_list, 
                                    fpn_channels=512, 
                                    num_classes=num_classes)

    def _load_resnet_weights(self, model):
        try:
            state_dict = torch.load('./resnet101-cd907fc2.pth', map_location='cpu')
            model.load_state_dict(state_dict)
            print("✅ Weight loaded for backbone")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}")

    def extract_features(self, backbone, x):
        """提取 ResNet 的四个阶段特征"""
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        
        l1 = backbone.layer1(x)
        l2 = backbone.layer2(l1)
        l3 = backbone.layer3(l2)
        l4 = backbone.layer4(l3)
        return [l1, l2, l3, l4]

    def forward(self, x):
        input_size = x.shape[2:]
        
        # 预处理：此处保留您原有的 CLAHE 增强逻辑（简化演示）
        if self.use_rgb:
            # 这里的增强逻辑应与您原代码一致，为了简洁此处不重复展开
            # 假设 x 已经是增强后的 tensor
            pass

        modal_feats = []
        if self.use_rgb: modal_feats.append(self.extract_features(self.rgb_backbone, x))
        if self.use_hsv: 
            hsv = self.rgb_to_hsv_batch(x)
            modal_feats.append(self.extract_features(self.hsv_backbone, hsv))
        if self.use_lab:
            lab = self.rgb_to_lab_batch(x)
            modal_feats.append(self.extract_features(self.lab_backbone, lab))

        # 将不同模态的同一层特征进行拼接
        # 例如 fusion_l1 = cat([rgb_l1, hsv_l1, lab_l1], dim=1)
        fused_features = []
        for i in range(4): # 对应 4 个 layer
            level_i_feats = [m[i] for m in modal_feats]
            fused_features.append(torch.cat(level_i_feats, dim=1))

        # UPerNet 解码
        logits = self.decode_head(fused_features)
        
        return F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)

    # 保留辅助函数
    def rgb_to_hsv_batch(self, rgb_tensor):
        # ... 原代码逻辑 ...
        return rgb_tensor # 占位

    def rgb_to_lab_batch(self, rgb_tensor):
        # ... 原代码逻辑 ...
        return rgb_tensor # 占位