import matplotlib.pyplot as plt

# EfficientNet-B3 的训练数据（10轮）
epochs = list(range(1, 11))
train_loss = [
    0.6605, 0.5764, 0.4786, 0.3612, 0.2900, 0.2084, 0.2091, 0.1543, 0.1037,
    0.0900
]
val_acc = [
    0.5734, 0.5874, 0.6503, 0.7413, 0.7972, 0.7622, 0.8462, 0.8392, 0.8392,
    0.8112
]
val_f1 = [
    0.0000, 0.0635, 0.2857, 0.5747, 0.7010, 0.6136, 0.7800, 0.7677, 0.7850,
    0.7379
]
val_kappa = [
    -0.0278, 0.0117, 0.1793, 0.4209, 0.5582, 0.4693, 0.6675, 0.6515, 0.6587,
    0.5952
]

# 绘图
plt.figure(figsize=(10, 5))

plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_acc, marker='s', label='Val Accuracy')
plt.plot(epochs, val_f1, marker='^', label='Val F1 Score')
plt.plot(epochs, val_kappa, marker='d', label='Val Kappa')

plt.title('EfficientNet-B3 Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
