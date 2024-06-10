import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

output_nums = 1

# 修改后的createXY函数
def createXY(dataset: pd.DataFrame, n_past: int, n_future: int, column_target: str):
    dataX, dataY = [], []
    for i in range(n_past, len(dataset) - n_future + 1):
        dataX.append(dataset.iloc[i - n_past:i].values)
        dataY.append(dataset.iloc[i:i + n_future][column_target].values)
    return np.array(dataX), np.array(dataY)

# 修改后的process_files函数
def process_files(columns_all, column_target, folder_path, n_past=1, n_future=1):
    all_dataX, all_dataY = np.array([]), np.array([])
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            series = pd.read_csv(file_path)
            single_dataset = series[columns_all]
            dataX, dataY = createXY(single_dataset, n_past, n_future, column_target)
            all_dataX = np.vstack([all_dataX, dataX]) if all_dataX.size else dataX
            all_dataY = np.vstack([all_dataY, dataY]) if all_dataY.size else dataY
    return all_dataX, all_dataY

# 'carbohydrate','protein','fat','cellulose'
# columns_all = ['CGM (mg / dl)','CSII - basal insulin (Novolin R, IU / H)','carbohydrate','protein','fat','cellulose']
columns_all = ['CGM (mg / dl)','CSII - basal insulin (Novolin R, IU / H)']
column_target = ['CGM (mg / dl)']
folder_path = './diabetes_datasets/T1'

# 使用n_past=8, n_future=4调用process_files
n_past = 8
dataX, dataY = process_files(columns_all, column_target, folder_path, n_past=n_past, n_future=output_nums)

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim
from torch.utils.data import Subset

# 假设dataX和dataY是你的数据
# 将它们转换为PyTorch张量，这里假设它们已经是Tensor或者从Numpy转换过来的
dataX_tensor = torch.tensor(dataX, dtype=torch.float32)
dataY_tensor = torch.tensor(dataY, dtype=torch.float32)

# 创建TensorDataset对象
dataset = TensorDataset(dataX_tensor, dataY_tensor)

# 其余拆分数据集、创建DataLoader对象的代码与之前相同
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - (train_size + val_size)

assert train_size + val_size + test_size == len(dataset)
# 按照顺序划分数据集
train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))
test_dataset = Subset(dataset, range(train_size + val_size, len(dataset)))

# 创建DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
feature_nums = dataX.shape[2]

# 修改后的LSTMRegressor类
class LSTMRegressor(nn.Module):
    def __init__(self, num_units=6, dropout=0.2, output_size=output_nums):
        super(LSTMRegressor, self).__init__()
        self.lstm1 = nn.LSTM(input_size=feature_nums, hidden_size=num_units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(in_features=num_units, out_features=output_size)
    
    def forward(self, X):
        X, _ = self.lstm1(X)
        X = self.dropout(X)
        X = X[:, -1, :]  # Get the last sequence output
        X = self.dense(X)
        return X

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 早停法参数
best_val_loss = float('inf')
patience = 10
patience_counter = 0
max_epochs = 500

# 训练过程
for epoch in range(max_epochs):
    train_loss = 0
    model.train()
    
    # Training loop
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    # 每个epoch后，在验证集上评估模型
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f'Epoch {epoch}: Training Loss: {train_loss}, Validation Loss: {val_loss}')

    # 早停法逻辑
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最好的模型状态
        best_model_state = model.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch} epochs.')
        model.load_state_dict(best_model_state)
        break

# 测试过程
model.eval()
test_loss = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss}')

# 加载最佳模型状态
model.load_state_dict(best_model_state)

# 在测试集上验证模型并输出预测标签
model.eval()
test_loss = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        all_predictions.append(output.cpu().numpy())
        all_targets.append(target.cpu().numpy())

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss}')

# 将预测标签和真实标签拼接成一个数组
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

all_predictions = all_predictions.reshape(-1)
all_targets = all_targets.reshape(-1)
all_predictions.shape, all_targets.shape

import matplotlib.pyplot as plt

# 画出比对图
plt.figure(figsize=(10, 5))
plt.plot(all_targets, label='True Labels', alpha=0.7)
plt.plot(all_predictions, label='Predicted Labels', alpha=0.7)
plt.legend()
plt.title('Comparison of True and Predicted Time Series')
plt.xlabel('Time')
plt.ylabel('Label Value')
plt.show()

