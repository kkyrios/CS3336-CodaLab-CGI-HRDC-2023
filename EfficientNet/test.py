import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from tqdm import tqdm


def load_model(weight_path, device):
    # model = EfficientNet.from_name('efficientnet-b0')
    model = EfficientNet.from_name('efficientnet-b3')
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img


def predict_image(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, pred = torch.max(output, 1)
        return int(pred.item())


def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
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


def run_inference(model_path, test_img_dir, label_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

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
        tensor_img = preprocess_image(img).to(device)
        pred = predict_image(model, tensor_img, device)
        results.append((img_name, pred))

    # 加载 groundtruth 并评估
    gt_df = pd.read_csv(label_file)
    gt_map = dict(zip(gt_df["Image"], gt_df["Hypertensive Retinopathy"]))
    y_true = [gt_map[name] for name, _ in results]
    y_pred = [pred for _, pred in results]

    evaluate_metrics(y_true, y_pred)


# ==== 请在此设置路径 ====
if __name__ == '__main__':
    model_path = "my/EfficientNet/efficientnet_retina.pth"
    test_img_dir = "task2_dataset/1-Images/1-Training Set"
    label_file = "task2_dataset/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
    run_inference(model_path, test_img_dir, label_file)
