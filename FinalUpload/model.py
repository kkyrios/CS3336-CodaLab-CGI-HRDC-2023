# model.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet


class ResNet34(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet34(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class model:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def load(self, dir_path):
        self.models = []

        # Load combined weights
        checkpoint = torch.load(os.path.join(dir_path, "model_weights.pth"),
                                map_location=self.device)

        # Load ResNet34
        self.resnet = ResNet34()
        self.resnet.load_state_dict(checkpoint['resnet'])
        self.resnet.to(self.device).eval()
        self.models.append(self.resnet)

        # Load DenseNet121
        self.densenet = models.densenet121(weights=None)
        self.densenet.classifier = nn.Linear(
            self.densenet.classifier.in_features, 2)
        self.densenet.load_state_dict(checkpoint['densenet'])
        self.densenet.to(self.device).eval()
        self.models.append(self.densenet)

        # Load EfficientNet-b0
        self.efficientnet = EfficientNet.from_name('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, 2)
        self.efficientnet.load_state_dict(checkpoint['efficientnet'])
        self.efficientnet.to(self.device).eval()
        self.models.append(self.efficientnet)

        # Transforms for EfficientNet
        self.eff_transform = lambda img: torch.from_numpy(
            ((cv2.resize(img[:, :, ::-1].astype(np.float32) / 255.0,
                         (224, 224)) - [0.485, 0.456, 0.406]) / [
                             0.229, 0.224, 0.225
                         ])).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Transform for DenseNet
        self.dense_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def crop_black_border(self, img, tol=7):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def apply_clahe_rgb(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def predict(self, input_image):
        with torch.no_grad():
            # ResNet34 preprocessing
            img_resnet = cv2.resize(input_image, (512, 512)) / 255.0
            tensor_resnet = torch.from_numpy(img_resnet).permute(
                2, 0, 1).unsqueeze(0).float().to(self.device)
            out_resnet = self.resnet(tensor_resnet)

            # DenseNet121 preprocessing
            dense_img = self.crop_black_border(input_image)
            dense_img = self.apply_clahe_rgb(dense_img)
            dense_img = cv2.cvtColor(dense_img, cv2.COLOR_BGR2RGB)
            dense_img = Image.fromarray(dense_img)
            tensor_densenet = self.dense_transform(dense_img).unsqueeze(0).to(
                self.device)
            out_densenet = self.densenet(tensor_densenet)

            # EfficientNet preprocessing
            tensor_efficientnet = self.eff_transform(input_image).float()
            out_efficientnet = self.efficientnet(tensor_efficientnet)

            # Weighted voting fusion (weights can be adjusted)
            weights = [0.4, 0.3, 0.3]
            outputs = [out_resnet, out_densenet, out_efficientnet]
            final_output = sum(w * o for w, o in zip(weights, outputs))
            _, pred = torch.max(final_output, 1)
            return int(pred.item())
