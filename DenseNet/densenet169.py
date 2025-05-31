import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import random


# ------------------ MixUp ------------------
def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ------------------ 图像预处理 ------------------
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


# ------------------ 自定义数据集 ------------------
class RetinaDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image"]
        label = int(self.df.loc[idx, "Hypertensive Retinopathy"])
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = preprocess_image(image)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


# ------------------ 数据增强 ------------------
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(),
    A.ShiftScaleRotate(shift_limit=0.05,
                       scale_limit=0.1,
                       rotate_limit=15,
                       p=0.5),
    A.OneOf([
        A.GaussianBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
    ],
            p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ------------------ 数据加载与采样 ------------------
csv_path = "task2_dataset/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
img_dir = "task2_dataset/1-Images/1-Training Set/"
df = pd.read_csv(csv_path)
print("总样本数:", len(df))
train_df, val_df = train_test_split(df,
                                    test_size=0.2,
                                    stratify=df["Hypertensive Retinopathy"],
                                    random_state=42)

class_counts = train_df["Hypertensive Retinopathy"].value_counts().to_dict()
print("训练集类别分布:", class_counts)
weights = train_df["Hypertensive Retinopathy"].apply(
    lambda x: 1.0 / class_counts[x]).values
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_dataset = RetinaDataset(train_df, img_dir, transform=train_transform)
val_dataset = RetinaDataset(val_df, img_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ------------------ 模型与训练 ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 patience=2,
                                                 factor=0.5)

history = {"loss": [], "val_acc": [], "val_f1": [], "val_kappa": []}


def validate_model(model, val_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            flips = [
                imgs,
                torch.flip(imgs, dims=[-1]),
                torch.flip(imgs, dims=[-2])
            ]
            outputs = torch.stack([model(f) for f in flips], dim=0).mean(dim=0)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    kappa = cohen_kappa_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    print(f"Val Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)
    return acc, f1, kappa


def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Validation Accuracy & F1')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['val_kappa'], label='Kappa')
    plt.title('Cohen Kappa')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


def train_model(model, train_loader, val_loader, num_epochs=20):
    best_kappa = 0
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{num_epochs} 开始...")
        model.train()
        train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
            outputs = model(imgs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                outputs, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"  Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
        acc, f1, kappa = validate_model(model, val_loader)
        scheduler.step(avg_loss)

        history["loss"].append(avg_loss)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        history["val_kappa"].append(kappa)

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), "best_densenet169.pth")
            print("✅ Saved Best Model")

    pd.DataFrame(history).to_csv("densenet169_training_history.csv",
                                 index=False)
    plot_history(history)


if __name__ == "__main__":
    print("🚀 开始训练模型 DenseNet169 + MixUp + TTA...")
    train_model(model, train_loader, val_loader, num_epochs=20)
