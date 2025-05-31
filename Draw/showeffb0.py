import matplotlib.pyplot as plt

# EfficientNet-B0 的训练数据（10轮）
epochs = list(range(1, 11))
train_loss = [
    0.6758, 0.6186, 0.5481, 0.4996, 0.4667, 0.3852, 0.3471, 0.2783, 0.2491,
    0.2235
]
val_acc = [
    0.5804, 0.6014, 0.6084, 0.6364, 0.6713, 0.6923, 0.7552, 0.7622, 0.7692,
    0.7902
]

# 绘图
plt.figure(figsize=(10, 5))

plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_acc, marker='s', label='Val Accuracy')

plt.title('EfficientNet-B0 Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
