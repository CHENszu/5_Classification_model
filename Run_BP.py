import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# 读取数据文件
file_path = input("请输入xlsx/csv文件路径：")
if file_path.endswith('.xlsx'):
    df = pd.read_excel(file_path)
elif file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
else:
    raise ValueError("仅支持xlsx或csv文件格式！")

# 处理标签列
label_col = df.columns[-1]
features = df.columns[:-1]

# 转换标签列为数值
le = LabelEncoder()
mask = df[label_col].notna()
df.loc[mask, label_col] = le.fit_transform(df.loc[mask, label_col].astype(str))
df[label_col] = pd.to_numeric(df[label_col], errors='coerce')

# 分割数据集
train_mask = df[label_col].notna()
predict_mask = df[label_col].isna()

X_train = df.loc[train_mask, features].values.astype(np.float32)
y_train = df.loc[train_mask, label_col].values.astype(np.int64)
X_predict = df.loc[predict_mask, features].values.astype(np.float32)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_predict_scaled = scaler.transform(X_predict) if len(X_predict) > 0 else None

# 转换为Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


# 神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 参数配置
config = {
    'input_size': X_train.shape[1],
    'hidden_size': 64,
    'output_size': len(le.classes_),
    'learning_rate': 0.001,
    'num_epochs': 200,
    'batch_size': 32
}

# 初始化模型
model = NeuralNet(config['input_size'], config['hidden_size'], config['output_size'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# 创建DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# 训练循环
model.train()
for epoch in range(config['num_epochs']):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 计算训练集准确率
    with torch.no_grad():
        outputs = model(X_train_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)
    print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], 准确率: {accuracy:.4f}')

# 进行预测
if len(X_predict) > 0:
    X_predict_tensor = torch.tensor(X_predict_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(X_predict_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels = le.inverse_transform(predicted.numpy())

    # 创建一个新的数据框仅包含需要预测的数据行
    predict_df = df.loc[predict_mask].copy()

    # 更新需要预测的标签列
    #predict_df[label_col] = predicted_labels
    predict_df[label_col] = predicted_labels.astype(np.int32)

    # 保存结果
    output_path = os.path.join(os.path.dirname(file_path), 'res_BP.xlsx')
    predict_df.to_excel(output_path, index=False)
    print(f"预测结果已保存至：{output_path}")
else:
    print("没有需要预测的数据")