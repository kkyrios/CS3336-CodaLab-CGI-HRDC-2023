import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


# 预处理函数：裁剪黑边
def crop_black_border(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


# 预处理函数：CLAHE增强
def apply_clahe_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# DenseNet 专用 transform
dense_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 读取一张训练图像路径
img_path = "task2_dataset/1-Images/1-Training Set/0000a5c9.png"  # 替换为你本地的一张图片
original = cv2.imread(img_path)

# 原图显示用 RGB
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# 应用预处理
processed = crop_black_border(original)
processed = apply_clahe_rgb(processed)
processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
processed_resized = cv2.resize(processed_rgb, (224, 224))  # 仅用于显示，不做标准化

# 可视化原图和处理后图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_resized)
plt.title("After Preprocessing")
plt.axis("off")

plt.tight_layout()
plt.show()
