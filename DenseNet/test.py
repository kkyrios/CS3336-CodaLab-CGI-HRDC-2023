import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from torchvision import models, transforms
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from tqdm import tqdm


# 加载模型
def load_model(weight_path, device):
    model = models.densenet169(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# CLAHE + 裁剪
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


# 图像预处理
def preprocess_image(img, transform):
    img = crop_black_border(img)
    img = apply_clahe_rgb(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)
    return img


# 单图预测
def predict_image(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, pred = torch.max(output, 1)
        return int(pred.item())


# 指标计算
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):  # 防止测试集中只有一个类别时出错
        print("Confusion Matrix Error: check input labels.")
        return
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    average = np.mean([f1, specificity, kappa])

    print("📊 Evaluation Metrics:")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Kappa        : {kappa:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"Average      : {average:.4f}")


# 主流程
def run_inference(model_path, test_img_dir, label_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    results = []
    image_files = sorted([
        f for f in os.listdir(test_img_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

    for img_name in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: {img_name} could not be read.")
            continue
        tensor_img = preprocess_image(img, transform).to(device)
        pred = predict_image(model, tensor_img, device)
        results.append((img_name, pred))

    # 加载 GroundTruth 并评估
    gt_df = pd.read_csv(label_file)
    gt_map = dict(zip(gt_df["Image"], gt_df["Hypertensive Retinopathy"]))
    y_true = [gt_map[name] for name, _ in results if name in gt_map]
    y_pred = [pred for name, pred in results if name in gt_map]

    evaluate_metrics(y_true, y_pred)


# 设置路径
if __name__ == '__main__':
    model_path = "my/DenseNet/best_densenet169.pth"
    test_img_dir = "task2_dataset/1-Images/1-Training Set"
    label_file = "task2_dataset/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
    run_inference(model_path, test_img_dir, label_file)
