import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class TestDataset(Dataset):

    def __init__(
        self,
        image_dir,
        use_rgb=True,
        use_hsv=False,
        use_lab=False
    ):

        self.image_dir = image_dir

        self.image_names = sorted(os.listdir(image_dir))

        self.use_rgb = use_rgb
        self.use_hsv = use_hsv
        self.use_lab = use_lab

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        channels = []

        # RGB
        if self.use_rgb:
            rgb = img.astype(np.float32) / 255.0
            channels.append(rgb)

        # HSV
        if self.use_hsv:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32) / 255.0
            channels.append(hsv)

        # LAB
        if self.use_lab:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0
            channels.append(lab)

        img = np.concatenate(channels, axis=2)

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, image_name