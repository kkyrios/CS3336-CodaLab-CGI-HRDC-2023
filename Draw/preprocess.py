import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np


# 图像预处理函数（去除黑边 + CLAHE增强）
def preprocess_image(image, use_clahe=True):

    def crop_black_border(img, tol=7):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def apply_clahe_rgb(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    image = crop_black_border(image)
    if use_clahe:
        image = apply_clahe_rgb(image)
    return image


# 自定义 Dataset
class HypertensionRetinaDataset(Dataset):

    def __init__(self, csv_path, image_dir, transform=None):
        self.labels_df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        label = int(self.labels_df.iloc[idx, 1])

        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = preprocess_image(image)  # 去黑边+CLAHE
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# 图像增强（训练）和标准化（验证）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化数据集
train_csv_path = "/Users/a1-6/Desktop/计算机视觉（强化）/大作业/task2_dataset/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
train_img_path = "/Users/a1-6/Desktop/计算机视觉（强化）/大作业/task2_dataset/1-Images/1-Training Set/"

# 示例：创建训练集
train_dataset = HypertensionRetinaDataset(train_csv_path,
                                          train_img_path,
                                          transform=train_transform)
train_loader = DataLoader(train_dataset,
                          batch_size=16,
                          shuffle=True,
                          num_workers=2)

# 示例：验证集（可划分数据集后使用）
# val_dataset = HypertensionRetinaDataset(train_csv_path, train_img_path, transform=val_transform)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ✅ 使用示例
if __name__ == "__main__":
    for images, labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Label batch:", labels)
        break
