import os
import cv2
import numpy as np

orig_label_dir = 'dataset/tra_pri/tra/label'
dirty_mask_dir = 'weights_sgfmb5_test2_2/dirty_data_analysis/dirty_pixel_masks'
fixed_label_dir = 'dataset/tra_pri/tra/label_cleaned'

os.makedirs(fixed_label_dir, exist_ok=True)

for fname in os.listdir(orig_label_dir):
    label = cv2.imread(os.path.join(orig_label_dir, fname), 0)
    
    # 找对应的 dirty mask
    base = os.path.splitext(fname)[0]
    dirty_path = os.path.join(dirty_mask_dir, base + '.png')
    
    if os.path.exists(dirty_path):
        dirty = cv2.imread(dirty_path, 0)
        # 把脏像素设为 255 (ignore)
        label[dirty > 0] = 255
        print(f"Cleaned {fname}: {(dirty > 0).sum()} pixels marked as ignore")
    
    cv2.imwrite(os.path.join(fixed_label_dir, fname), label)

print(f"Done! Cleaned labels saved to: {fixed_label_dir}")