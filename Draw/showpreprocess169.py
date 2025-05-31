import matplotlib.pyplot as plt

# DenseNet169 的训练数据（20轮）
epochs = list(range(1, 21))
train_loss = [
    0.6736, 0.6099, 0.5799, 0.5390, 0.5279, 0.5294, 0.4910, 0.5085, 0.5203,
    0.4630, 0.5035, 0.4238, 0.3841, 0.4009, 0.4202, 0.3883, 0.3698, 0.3772,
    0.3926, 0.2980
]
val_acc = [
    0.7483, 0.7273, 0.7133, 0.7762, 0.7413, 0.7552, 0.7413, 0.7972, 0.8601,
    0.7203, 0.8042, 0.8252, 0.8322, 0.8252, 0.8392, 0.8182, 0.8322, 0.8462,
    0.8811, 0.8462
]
val_f1 = [
    0.6842, 0.6214, 0.4810, 0.6863, 0.7338, 0.7482, 0.7448, 0.7786, 0.8413,
    0.7297, 0.7586, 0.8000, 0.8154, 0.8062, 0.8189, 0.8060, 0.8182, 0.8281,
    0.8547, 0.8308
]
val_kappa = [
    0.4753, 0.4152, 0.3440, 0.5189, 0.4931, 0.5205, 0.5002, 0.5949, 0.7172,
    0.4635, 0.5940, 0.6456, 0.6639, 0.6491, 0.6755, 0.6394, 0.6656, 0.6904,
    0.7541, 0.6919
]

# 设置图形风格
plt.figure(figsize=(10, 5))

# 绘制每个指标的变化曲线
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_acc, marker='s', label='Val Accuracy')
plt.plot(epochs, val_f1, marker='^', label='Val F1 Score')
plt.plot(epochs, val_kappa, marker='d', label='Val Kappa')

# 图形设置
plt.title('DenseNet169 Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图像
plt.show()
