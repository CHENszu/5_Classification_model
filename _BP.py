import torch
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def run_bp(X_train, y_train, X_predict, config=None):
    # 默认参数配置
    default_config = {
        'hidden_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 200,
        'batch_size': 32,
        'val_size': 0.2,
        'random_state': 42
    }
    config = config or {}
    config = {**default_config, **config}

    # 划分验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=config['val_size'],
        random_state=config['random_state']
    )

    # 转换为Tensor并创建DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train_split, dtype=torch.float32),
        torch.tensor(y_train_split, dtype=torch.long)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    # 初始化模型
    model = NeuralNet(
        input_size=X_train.shape[1],
        hidden_size=config['hidden_size'],
        output_size=len(np.unique(y_train))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    for epoch in range(config['num_epochs']):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 验证集评估
    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        outputs = model(val_tensor)
        _, predicted = torch.max(outputs, 1)
        val_accuracy = (predicted == torch.tensor(y_val)).sum().item() / len(y_val)

    # 预测概率
    predict_proba = None
    if len(X_predict) > 0:
        with torch.no_grad():
            outputs = model(torch.tensor(X_predict, dtype=torch.float32))
            predict_proba = torch.softmax(outputs, dim=1).numpy()

    return {
        'name': 'BP_Network',
        'proba': predict_proba,
        'val_accuracy': val_accuracy,
        'config': config
    }