# models/mask2segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
import cv2

# =========================
# 1. 颜色空间转换模块
# =========================
class ColorSpaceConverter(nn.Module):
    """
    根据配置将RGB图像转换为HSV或LAB，并与RGB拼接。
    """
    def __init__(self, use_rgb=True, use_hsv=False, use_lab=False):
        super().__init__()
        self.use_rgb = use_rgb
        self.use_hsv = use_hsv
        self.use_lab = use_lab

        self.out_channels = 0
        if use_rgb:
            self.out_channels += 3
        if use_hsv:
            self.out_channels += 3
        if use_lab:
            self.out_channels += 3

    def forward(self, x):
        # x: (B, 3, H, W) in range [0, 1]
        outputs = []

        if self.use_rgb:
            outputs.append(x)

        x_np = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        converted = []

        for img in x_np:
            img_list = []
            if self.use_hsv:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hsv = torch.from_numpy(hsv).permute(2, 0, 1).float() / 255.0
                img_list.append(hsv)
            if self.use_lab:
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                lab = torch.from_numpy(lab).permute(2, 0, 1).float() / 255.0
                img_list.append(lab)
            if img_list:
                converted.append(torch.cat(img_list, dim=0))

        if converted:
            converted = torch.stack(converted).to(x.device)
            outputs.append(converted)

        return torch.cat(outputs, dim=1)


# =========================
# 2. Backbone: ResNet
# =========================
class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=3, backbone='resnet101', pretrained=True):
        super().__init__()

        if backbone == 'resnet50':
            net = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            channels = [256, 512, 1024, 2048]
        else:
            net = resnet101(weights="IMAGENET1K_V1" if pretrained else None)
            channels = [256, 512, 1024, 2048]

        # 修改第一层以适配多通道输入
        if in_channels != 3:
            net.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.channels = channels

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


# =========================
# 3. Pixel Decoder (FPN)
# =========================
class PixelDecoder(nn.Module):
    """
    简化版 FPN，用于融合多尺度特征。
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for c in in_channels[::-1]:
            self.lateral_convs.append(nn.Conv2d(c, out_channels, 1))
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        results = []
        x = None
        for i, feat in enumerate(features[::-1]):
            lateral = self.lateral_convs[i](feat)
            if x is None:
                x = lateral
            else:
                x = lateral + F.interpolate(
                    x, size=lateral.shape[-2:], mode='bilinear', align_corners=False
                )
            x = self.output_convs[i](x)
            results.append(x)

        return results[-1]  # 返回最高分辨率特征


# =========================
# 4. Transformer Decoder
# =========================
class TransformerDecoder(nn.Module):
    """
    Mask2Former 风格的 Transformer 解码器。
    """
    def __init__(self, hidden_dim=256, num_queries=100, num_classes=4, num_layers=6):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
        self.mask_embed = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pixel_features):
        B, C, H, W = pixel_features.shape

        # Flatten spatial features
        memory = pixel_features.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        tgt = torch.zeros_like(queries)

        hs = self.decoder(tgt, memory)

        class_logits = self.class_embed(hs)        # (B, Q, num_classes+1)
        mask_embed = self.mask_embed(hs)           # (B, Q, C)

        # 生成 mask
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_features)

        return class_logits, masks


# =========================
# 5. Mask2Segmentation 主模型
# =========================
class Mask2Segmentation(nn.Module):
    """
    可替换 MultiModalUPerNet 的 Mask2Former 语义分割模型。
    """
    def __init__(
        self,
        num_classes=4,
        use_rgb=True,
        use_hsv=False,
        use_lab=False,
        backbone='resnet101',
        num_queries=100
    ):
        super().__init__()

        # 颜色空间转换
        self.color_converter = ColorSpaceConverter(
            use_rgb, use_hsv, use_lab
        )
        in_channels = self.color_converter.out_channels

        # Backbone
        self.backbone = ResNetBackbone(
            in_channels=in_channels,
            backbone=backbone
        )

        # Pixel Decoder
        self.pixel_decoder = PixelDecoder(self.backbone.channels)

        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(
            hidden_dim=256,
            num_queries=num_queries,
            num_classes=num_classes
        )

        self.num_classes = num_classes

    def forward(self, x):
        input_size = x.shape[-2:]

        # 颜色空间转换
        x = self.color_converter(x)

        # Backbone 特征
        features = self.backbone(x)

        # Pixel Decoder
        pixel_features = self.pixel_decoder(features)

        # Transformer Decoder
        class_logits, mask_logits = self.transformer_decoder(pixel_features)

        # 语义分割输出
        class_probs = F.softmax(class_logits, dim=-1)[..., :-1]  # 去掉 no-object
        masks = torch.sigmoid(mask_logits)

        # 加权融合得到语义分割图
        seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, masks)

        # 上采样到原始尺寸
        seg_logits = F.interpolate(
            seg_logits,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )

        return seg_logits