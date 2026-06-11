import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
# 导入并列的两个核心前处理函数
from .processors import extract_foreground_with_removal, extract_foreground_with_blur_bg

class AluminiumDataset(Dataset):
    """
    Custom Dataset for Aluminium Scrap Classification.
    Directly reflects the first part of the experiment: 'none' vs 'blur' vs 'removal'.
    """
    def __init__(self, root_dir, background_path, bg_mode='none', transform=None):
        self.root_dir = root_dir
        self.background_path = background_path
        self.bg_mode = bg_mode  # 'none', 'blur', 'removal'
        self.transform = transform
        
        self.classes = ['Cast_Alu_Images', 'Wrought_Alu_Images']
        self.image_paths = []
        self.labels = []

        # Read background image into memory once to save I/O time
        self.back_img_bgr = cv2.imread(background_path)

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 1. Load current image via OpenCV for processing
        img_bgr = cv2.imread(img_path)

        # 2. Apply the selected experiment baseline
        if self.bg_mode == 'removal':
            img_processed = extract_foreground_with_removal(img_bgr, self.back_img_bgr)
        elif self.bg_mode == 'blur':
            img_processed = extract_foreground_with_blur_bg(img_bgr, self.back_img_bgr)
        else:
            img_processed = img_bgr  # 'none' mode: use original image

        # 3. Convert BGR back to PIL RGB for torchvision compatibility
        img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        # 4. Standard geometric transforms and normalization
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)