import matplotlib.pyplot as plt

# 定义每轮的训练与验证数据
epochs = list(range(1, 11))
train_loss = [
    0.6494, 0.5074, 0.3870, 0.3177, 0.2415, 0.1821, 0.1390, 0.1373, 0.1799,
    0.0887
]
val_acc = [
    0.6224, 0.7133, 0.7413, 0.7832, 0.8671, 0.8252, 0.8392, 0.8322, 0.8601,
    0.8671
]
val_f1 = [
    0.1818, 0.5393, 0.7376, 0.6869, 0.8480, 0.8062, 0.8034, 0.7736, 0.8333,
    0.8288
]
val_kappa = [
    0.1033, 0.3618, 0.4955, 0.5303, 0.7306, 0.6491, 0.6673, 0.6429, 0.7129,
    0.7210
]

# 设置图形风格
plt.figure(figsize=(8, 4))

# 画出每个指标的变化曲线
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_acc, marker='s', label='Val Accuracy')
plt.plot(epochs, val_f1, marker='^', label='Val F1 Score')
plt.plot(epochs, val_kappa, marker='d', label='Val Kappa')

# 图形设置
plt.title('DenseNet121 Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图像
plt.show()
