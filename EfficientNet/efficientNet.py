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
from sklearn.metrics import accuracy_score, confusion_matrix


# ========== 1. 图像预处理 ==========
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


# ========== 2. 自定义 Dataset ==========
class RetinaDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = preprocess_image(image)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# ========== 3. 数据增强 ==========
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

# ========== 4. 加载数据 ==========
csv_path = "/Users/a1-6/Desktop/计算机视觉（强化）/大作业/task2_dataset/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
img_dir = "/Users/a1-6/Desktop/计算机视觉（强化）/大作业/task2_dataset/1-Images/1-Training Set/"

full_dataset = RetinaDataset(csv_path, img_dir, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ========== 5. 模型定义 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ========== 6. 训练函数 ==========
def train_model(model, train_loader, val_loader, num_epochs=10):
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
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}")
        validate_model(model, val_loader)


# ========== 7. 验证函数 ==========
def validate_model(model, val_loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.numpy())

    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)


# ========== 8. 启动训练 ==========
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, num_epochs=10)
    torch.save(model.state_dict(), "efficientnet_retina.pth")


# ========== 9. 加载模型进行测试 ==========
def test_model(model_path, test_loader):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.numpy())

    acc = accuracy_score(trues, preds)
    print(f"Test Accuracy: {acc:.4f}")
