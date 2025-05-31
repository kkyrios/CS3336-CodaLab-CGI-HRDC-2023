# ========== 导入所需库 ==========
import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score


# ========== 图像预处理函数 ==========
# 包括黑边裁剪和CLAHE对比度增强
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


# ========== 自定义数据集类 ==========
class RetinaDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取图像路径和标签
        img_name = self.df.loc[idx, "Image"]
        label = int(self.df.loc[idx, "Hypertensive Retinopathy"])
        img_path = os.path.join(self.image_dir, img_name)

        # 图像读取与预处理
        image = cv2.imread(img_path)
        image = preprocess_image(image)
        image = cv2.resize(image, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# ========== 数据增强策略 ==========
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 光照增强
    transforms.RandomResizedCrop(300, scale=(0.9, 1.0)),  # 随机裁剪缩放
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((300, 300)),  # 验证集统一尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 数据路径配置 ==========
csv_path = "/Users/a1-6/Desktop/计算机视觉（强化）/大作业/task2_dataset/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
img_dir = "/Users/a1-6/Desktop/计算机视觉（强化）/大作业/task2_dataset/1-Images/1-Training Set/"

# ========== 分层划分训练集与验证集 ==========
df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df,
                                    test_size=0.2,
                                    stratify=df["Hypertensive Retinopathy"],
                                    random_state=42)

train_dataset = RetinaDataset(train_df, img_dir, transform=train_transform)
val_dataset = RetinaDataset(val_df, img_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ========== 模型定义 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
model = model.to(device)

# 损失函数、优化器与学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 patience=2,
                                                 factor=0.5)


# ========== 验证函数 ==========
def validate_model(model, val_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    # 计算指标
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    kappa = cohen_kappa_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    print(f"Val Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)
    return acc, f1, kappa


# ========== 训练函数 ==========
def train_model(model, train_loader, val_loader, num_epochs=10):
    best_kappa = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        acc, f1, kappa = validate_model(model, val_loader)
        scheduler.step(avg_loss)

        # 保存最佳模型（基于Kappa）
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), "best_efficientnet_b3.pth")
            print("✅ Saved Best Model")


# ========== 程序入口 ==========
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, num_epochs=10)
