import pandas as pd
import matplotlib.pyplot as plt

# 加载训练日志
df = pd.read_csv("densenet121_training_history.csv")

# 绘图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df['loss'], label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['val_acc'], label='Val Acc')
plt.plot(df['val_f1'], label='Val F1')
plt.plot(df['val_kappa'], label='Val Kappa')
plt.title("Validation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("densenet121_training_curve.png")
plt.show()
