import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

class Mask2FormerHF(nn.Module):
    """
    使用 HuggingFace 预训练权重的 Mask2Former，
    输出格式与 MultiModalUPerNet 兼容：(B, num_classes, H, W)
    """
    def __init__(
        self,
        num_classes=4,
        pretrained_model="facebook/mask2former-swin-small-ade-semantic"
    ):
        super().__init__()

        # 加载预训练模型
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x: Tensor (B, 3, H, W) in range [0, 1]
        Returns:
            seg_logits: (B, num_classes, H, W)
        """
        outputs = self.model(pixel_values=x)

        # 获取 mask logits 和类别 logits
        class_queries_logits = outputs.class_queries_logits  # (B, Q, C+1)
        masks_queries_logits = outputs.masks_queries_logits  # (B, Q, H, W)

        # 去掉 no-object 类
        class_probs = torch.softmax(class_queries_logits, dim=-1)[..., :-1]
        masks_probs = torch.sigmoid(masks_queries_logits)

        # Mask2Former 的语义融合方式
        seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)

        # 上采样到输入尺寸
        seg_logits = F.interpolate(
            seg_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return seg_logits