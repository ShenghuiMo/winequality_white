
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

wine_path = "wine_data//winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=';', skiprows=1)
wine_tensor = torch.from_numpy(wine_numpy)
data = wine_tensor[:, :-1]
print(data.shape)

# 归一化示例（min-max 或 z-score 标准化）
for i in range(0, data.shape[1]):
    data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()

target = wine_tensor[:, -1]
print(target.shape)
# 手动构建Batch数据集
train_data = data[:4800, :]    # 训练集 4800
val_data = data[4800:4896, :]  # 验证集 96
train_target = target[:4800]
val_target = target[4800:4896]
batch_number = 32
batch_train_data = train_data.reshape(-1, batch_number, 11)
batch_val_data = val_data.reshape(-1, batch_number, 11)
batch_target_data = train_target.reshape(-1, batch_number, 1)
batch_val_target = val_target.reshape(-1, batch_number, 1)
print(train_data.shape, val_data.shape, train_target.shape, val_target.shape, batch_train_data.shape,
      batch_target_data.shape, batch_val_data.shape, batch_val_target.shape)  # 测试张量的形状大小

# 构建网络
model = nn.Sequential(
    nn.Linear(11, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
print(model) # 查看网络信息

# 构建优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # 使用adam优化器，并且采用正则化防止过拟合


# 构建训练器
def train_loop(epoch, loss_function, optimizer, batch_train_data, batch_target_data, batch_val_data, batch_val_target):
    train_losses = []
    val_losses = []

    for e in range(1, epoch + 1):
        model.train()
        epoch_train_loss = 0.0

        for d in range(batch_train_data.shape[0]):
            pred = model(batch_train_data[d])  # [4, 1]
            loss = loss_function(pred, batch_target_data[d])  # [4, 1] vs [4, 1]
            optimizer.zero_grad()
            loss.backward()                       # 计算梯度，并将梯度保存在参数的属性中
            optimizer.step()                      # 更新参数
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / batch_train_data.shape[0]
        train_losses.append(avg_train_loss)

        model.eval()                              # 启动model.eval()模式关闭梯度
        epoch_val_loss = 0.0
        with torch.no_grad():
            for v in range(batch_val_data.shape[0]):
                pred = model(batch_val_data[v])
                loss = loss_function(pred, batch_val_target[v])
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / batch_val_data.shape[0]
        val_losses.append(avg_val_loss)

        print(f"Epoch {e}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    return train_losses, val_losses


train_losses, val_losses = train_loop(
    200,
    loss_function=nn.MSELoss(),
    optimizer=optimizer,
    batch_train_data=batch_train_data,
    batch_target_data=batch_target_data,
    batch_val_data=batch_val_data,
    batch_val_target=batch_val_target
)
# 每轮训练的折线图可视化
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

model.eval()
# 每轮在测试集上预测值和真实值的差异可视化
predictions = []
ground_truth = []

with torch.no_grad():
    for v in range(batch_val_data.shape[0]):
        pred = model(batch_val_data[v])  # 输出形状: [4, 1]
        predictions.append(pred.squeeze().numpy())  # 去掉多余维度，转为 NumPy
        ground_truth.append(batch_val_target[v].numpy())

# 展平为一维数组
predictions = np.concatenate(predictions)
ground_truth = np.concatenate(ground_truth)
plt.figure(figsize=(10, 5))
plt.plot(ground_truth, label='True Quality', marker='o')
plt.plot(predictions, label='Predicted Quality', marker='x')
plt.title("Wine Quality Prediction: True vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Wine Quality")
plt.legend()
plt.grid(True)
plt.show()

# 预测和真实值之间的散点图可视化
plt.figure(figsize=(6, 6))
plt.scatter(ground_truth, predictions, c='blue', alpha=0.6)
plt.plot([min(ground_truth), max(ground_truth)],
         [min(ground_truth), max(ground_truth)],
         'r--', label='Perfect Prediction')
plt.xlabel("True Quality")
plt.ylabel("Predicted Quality")
plt.title("Scatter: Prediction vs True")
plt.legend()
plt.grid(True)
plt.show()
