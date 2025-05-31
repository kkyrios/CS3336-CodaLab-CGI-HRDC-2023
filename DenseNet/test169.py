import os
import cv2
import torch
import pandas as pd
from model import model
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

# 数据路径配置
csv_path = "task2_dataset/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
img_dir = "task2_dataset/1-Images/1-Training Set/"

# 加载标签
df = pd.read_csv(csv_path)

# 加载模型
net = model()
net.load(".")  # 当前路径下应该有 model_weights.pth

# 推理
y_true, y_pred = [], []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
    img_path = os.path.join(img_dir, row["Image"])
    label = int(row["Hypertensive Retinopathy"])
    img = cv2.imread(img_path)
    pred = net.predict(img)
    y_true.append(label)
    y_pred.append(pred)

# 评估
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"Accuracy     : {acc:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"Cohen Kappa  : {kappa:.4f}")
print("Confusion Matrix:")
print(cm)
