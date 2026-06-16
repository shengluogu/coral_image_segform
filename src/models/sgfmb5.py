import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class MultiModalSegFormerB5(nn.Module):
    """
    Multi-Modal SegFormer-B5 for Semantic Segmentation.
    Supports RGB, HSV, and LAB color spaces similar to the original
    MultiModalDeepLabV3Plus interface.
    """

    def __init__(
        self,
        num_classes=4,
        use_rgb=True,
        use_hsv=False,
        use_lab=False,
        pretrained=True
    ):
        super(MultiModalSegFormerB5, self).__init__()

        self.use_rgb = use_rgb
        self.use_hsv = use_hsv
        self.use_lab = use_lab

        # 计算输入通道数
        self.in_channels = 0
        if use_rgb:
            self.in_channels += 3
        if use_hsv:
            self.in_channels += 3
        if use_lab:
            self.in_channels += 3

        if self.in_channels == 0:
            raise ValueError("At least one color space must be enabled.")

        # ====== 加载预训练的 SegFormer-B5 ======
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640",
                num_labels=num_classes,
				use_safetensors=True,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig(
                num_labels=num_classes,
                depths=[3, 6, 40, 3],
                hidden_sizes=[64, 128, 320, 512],
                decoder_hidden_size=768,
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                mlp_ratios=[4, 4, 4, 4],
                drop_rate=0.1,)
            self.model = SegformerForSemanticSegmentation(config)

        # ====== 修改输入通道数以适配多模态 ======
        # SegFormer 的第一层是 patch embedding 卷积
        original_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        self._modify_input_conv(original_conv)

        # ====== 调整分类头 ======
        self.model.decode_head.classifier = nn.Conv2d(
            self.model.config.decoder_hidden_size,
            num_classes,
            kernel_size=1
        )

    def _modify_input_conv(self, original_conv):
        """修改第一层卷积以适配多模态输入，并继承预训练权重"""
        new_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )

        with torch.no_grad():
            if self.in_channels == 3:
                new_conv.weight.copy_(original_conv.weight)
            else:
                # 将原始权重复制到新的通道，并进行归一化
                repeat = self.in_channels // 3
                new_weight = original_conv.weight.repeat(1, repeat, 1, 1)
                new_weight = new_weight / repeat
                new_conv.weight.copy_(new_weight[:, :self.in_channels, :, :])

            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes, H, W)
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # 上采样到输入分辨率
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return logits