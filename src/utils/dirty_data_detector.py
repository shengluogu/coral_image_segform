# utils/dirty_data_detector.py
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict


class DirtyPixelDetector:
    """
    像素级脏数据检测器
    
    工作原理:
    1. 训练过程中，在指定的 epoch 窗口内累积每个像素的 per-pixel CE loss (EMA)
    2. 训练到指定 epoch 时, 按【类别】取 Top-K% 高 loss 像素作为脏像素
    3. 导出: 每张图的脏像素 mask + 图像级排序 CSV + 类别 loss 统计 JSON
    """

    def __init__(
        self,
        save_dir,
        num_classes=4,
        ignore_index=255,
        start_epoch=90,             # 开始累积 EMA 的 epoch
        end_epoch=200,              # 导出 dirty mask 的 epoch
        ema_alpha=0.1,              # EMA 平滑系数 (越大越看重最新)
        topk_percent_per_class=2.0, # 每个类别取 loss 最高的 Top-K% 像素
        class_names=None,
    ):
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.ema_alpha = ema_alpha
        self.topk_percent = topk_percent_per_class
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # 每张图的 EMA loss map: {filename: np.ndarray (H, W) float32}
        self.ema_loss_maps = {}
        # 每张图的 GT mask（用于按类别筛选时找像素位置）
        self.gt_masks = {}
        # 已累积的 epoch 计数（用于初始化 EMA）
        self.update_count = defaultdict(int)

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "dirty_pixel_masks"), exist_ok=True)

    def is_in_window(self, epoch):
        """判断当前 epoch 是否在统计窗口内 (1-based epoch)"""
        return self.start_epoch <= epoch <= self.end_epoch

    def should_export(self, epoch):
        return epoch == self.end_epoch

    @torch.no_grad()
    def update(self, pred_logits, masks, filenames):
        """
        每个 batch 调用一次, 累积 EMA loss
        
        Args:
            pred_logits: (B, C, H, W) 模型输出 logits
            masks:      (B, H, W)    GT 标签
            filenames:  list[str]    本 batch 每张图的文件名
        """
        # 计算 unweighted per-pixel CE (保证类别公平)
        # 输出 shape: (B, H, W)
        per_pixel_ce = F.cross_entropy(
            pred_logits, masks,
            reduction='none',
            ignore_index=self.ignore_index
        )

        # 把 ignore 区域的 loss 强制设为 0（cross_entropy 已经会置 0，这里保险）
        valid_mask = (masks != self.ignore_index)
        per_pixel_ce = per_pixel_ce * valid_mask.float()

        per_pixel_ce_np = per_pixel_ce.detach().cpu().numpy().astype(np.float32)
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)

        for i, fname in enumerate(filenames):
            loss_map = per_pixel_ce_np[i]  # (H, W)
            gt = masks_np[i]               # (H, W)

            if fname not in self.ema_loss_maps:
                # 首次出现该图，直接赋值
                self.ema_loss_maps[fname] = loss_map.copy()
                self.gt_masks[fname] = gt.copy()
            else:
                # EMA 更新
                prev = self.ema_loss_maps[fname]
                self.ema_loss_maps[fname] = (
                    (1 - self.ema_alpha) * prev + self.ema_alpha * loss_map
                )
                # GT 理论上不变，但每次覆盖一次保证一致
                self.gt_masks[fname] = gt

            self.update_count[fname] += 1

    def export(self, logger=None):
        """
        在 end_epoch 调用，导出脏像素 mask 和报告
        """
        def _log(msg):
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        _log(f"[DirtyDetector] Exporting results. Total images tracked: {len(self.ema_loss_maps)}")

        # ===== 第一步：按类别收集所有像素的 loss，确定每个类别的 Top-K 阈值 =====
        class_loss_pool = {c: [] for c in range(self.num_classes)}
        for fname, loss_map in self.ema_loss_maps.items():
            gt = self.gt_masks[fname]
            for c in range(self.num_classes):
                cls_pixels = loss_map[gt == c]
                if cls_pixels.size > 0:
                    class_loss_pool[c].append(cls_pixels)

        class_thresholds = {}
        class_stats = {}
        for c in range(self.num_classes):
            if len(class_loss_pool[c]) == 0:
                class_thresholds[c] = float('inf')
                class_stats[self.class_names[c]] = {"count": 0}
                continue
            all_losses = np.concatenate(class_loss_pool[c])
            # 取最高的 topk_percent% 作为脏像素阈值
            threshold = float(np.percentile(all_losses, 100 - self.topk_percent))
            class_thresholds[c] = threshold
            class_stats[self.class_names[c]] = {
                "count": int(all_losses.size),
                "mean_loss": float(all_losses.mean()),
                "median_loss": float(np.median(all_losses)),
                "p90_loss": float(np.percentile(all_losses, 90)),
                "p95_loss": float(np.percentile(all_losses, 95)),
                "p99_loss": float(np.percentile(all_losses, 99)),
                "threshold_used": threshold,
            }
            _log(f"[DirtyDetector] Class '{self.class_names[c]}': "
                 f"threshold (top{self.topk_percent}%) = {threshold:.4f}")

        # ===== 第二步：为每张图生成 dirty mask，并统计图像级指标 =====
        image_ranking = []  # [(fname, dirty_ratio, mean_loss, per_class_dirty_count)]
        mask_dir = os.path.join(self.save_dir, "dirty_pixel_masks")

        for fname, loss_map in self.ema_loss_maps.items():
            gt = self.gt_masks[fname]
            dirty_mask = np.zeros_like(gt, dtype=np.uint8)  # 0=clean, 1=dirty

            per_class_dirty = {}
            for c in range(self.num_classes):
                cls_region = (gt == c)
                if not cls_region.any():
                    per_class_dirty[self.class_names[c]] = 0
                    continue
                dirty_in_cls = cls_region & (loss_map > class_thresholds[c])
                dirty_mask[dirty_in_cls] = 1
                per_class_dirty[self.class_names[c]] = int(dirty_in_cls.sum())

            # 保存 dirty mask (0/1, 用 0/255 存方便查看)
            out_path = os.path.join(mask_dir, fname.replace('.jpg', '.png').replace('.jpeg', '.png'))
            # 保证扩展名为 png
            if not out_path.endswith('.png'):
                out_path = os.path.splitext(out_path)[0] + '.png'
            cv2.imwrite(out_path, dirty_mask * 255)

            valid = (gt != self.ignore_index)
            n_valid = int(valid.sum())
            n_dirty = int(dirty_mask.sum())
            dirty_ratio = n_dirty / (n_valid + 1e-10)
            mean_loss = float(loss_map[valid].mean()) if n_valid > 0 else 0.0

            image_ranking.append({
                "filename": fname,
                "dirty_ratio": dirty_ratio,
                "mean_loss": mean_loss,
                "dirty_pixels": n_dirty,
                "valid_pixels": n_valid,
                **{f"dirty_{k}": v for k, v in per_class_dirty.items()},
            })

        # ===== 第三步：图像级排序 CSV =====
        image_ranking.sort(key=lambda x: x["dirty_ratio"], reverse=True)
        csv_path = os.path.join(self.save_dir, "dirty_images_ranking.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            keys = list(image_ranking[0].keys())
            f.write(",".join(keys) + "\n")
            for row in image_ranking:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
        _log(f"[DirtyDetector] Image ranking saved to: {csv_path}")

        # ===== 第四步：类别 loss 统计 JSON =====
        stats_path = os.path.join(self.save_dir, "per_class_loss_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "start_epoch": self.start_epoch,
                    "end_epoch": self.end_epoch,
                    "ema_alpha": self.ema_alpha,
                    "topk_percent_per_class": self.topk_percent,
                },
                "per_class_stats": class_stats,
                "total_images": len(self.ema_loss_maps),
            }, f, indent=2, ensure_ascii=False)
        _log(f"[DirtyDetector] Per-class stats saved to: {stats_path}")
        _log(f"[DirtyDetector] Dirty pixel masks saved to: {mask_dir}")
        _log(f"[DirtyDetector] Top 10 dirtiest images:")
        for r in image_ranking[:10]:
            _log(f"  {r['filename']}: dirty_ratio={r['dirty_ratio']:.4f}, "
                 f"mean_loss={r['mean_loss']:.4f}, dirty_pixels={r['dirty_pixels']}")
