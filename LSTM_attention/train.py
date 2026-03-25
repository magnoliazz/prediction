import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from model import AttentionLSTM
from sklearn.preprocessing import StandardScaler
import joblib
import os

class OilPumpTemperaturePredictor:
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, lr=0.001, online_lr=0.0001):
        self.model = AttentionLSTM(input_size, hidden_size, num_layers, dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.online_optimizer = optim.Adam(self.model.parameters(), lr=online_lr)
        self.criterion = torch.nn.MSELoss()
        self.scaler = StandardScaler()
        self.input_size = input_size
        self.data_buffer = []  # 用于存储最近的W个时间步数据
        self.seq_len = 0  # 滑动窗口长度
    
    def preprocess_data(self, data, target_col, seq_len=10):
        # 提取特征和目标
        features = data.drop(columns=[target_col]).values
        target = data[target_col].values
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 构建序列数据
        X, y = [], []
        for i in range(len(features_scaled) - seq_len):
            X.append(features_scaled[i:i+seq_len])
            y.append(target[i+seq_len])
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, epochs=100, batch_size=32):
        self.model.train()
        
        # 转换为tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # 训练循环
        for epoch in range(epochs):
            # 随机打乱数据
            permutation = torch.randperm(X_tensor.size(0))
            X_tensor = X_tensor[permutation]
            y_tensor = y_tensor[permutation]
            
            total_loss = 0
            for i in range(0, X_tensor.size(0), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/(X_tensor.size(0)/batch_size):.4f}')
    
    def incremental_learn(self, new_data, target_col, seq_len=10, epochs=10, batch_size=16):
        self.model.train()
        
        # 提取特征和目标
        features = new_data.drop(columns=[target_col]).values
        target = new_data[target_col].values
        
        # 标准化特征（使用已有的scaler）
        features_scaled = self.scaler.transform(features)
        
        # 构建序列数据
        X, y = [], []
        for i in range(len(features_scaled) - seq_len):
            X.append(features_scaled[i:i+seq_len])
            y.append(target[i+seq_len])
        
        if len(X) == 0:
            print('Insufficient data for incremental learning')
            return
        
        # 转换为tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # 训练循环
        for epoch in range(epochs):
            permutation = torch.randperm(X_tensor.size(0))
            X_tensor = X_tensor[permutation]
            y_tensor = y_tensor[permutation]
            
            total_loss = 0
            for i in range(0, X_tensor.size(0), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'Incremental Learning Epoch [{epoch+1}/{epochs}], Loss: {total_loss/(X_tensor.size(0)/batch_size):.4f}')
    
    def predict(self, data, seq_len=10):
        self.model.eval()
        
        # 提取特征
        features = data.values
        
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        
        # 构建序列
        X = []
        if len(features_scaled) >= seq_len:
            X.append(features_scaled[-seq_len:])
        else:
            # 如果数据不足seq_len，用前面的数据填充
            padding = np.zeros((seq_len - len(features_scaled), self.input_size))
            X.append(np.vstack([padding, features_scaled]))
        
        # 转换为tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 预测
        with torch.no_grad():
            output = self.model(X_tensor)
        
        return output.item()
    
    def init_buffer(self, seq_len):
        """初始化数据缓冲区，设置滑动窗口长度"""
        self.seq_len = seq_len
        self.data_buffer = []
    
    def update_buffer(self, new_data):
        """更新数据缓冲区，保持最近的seq_len个时间步数据"""
        # 提取特征数据
        if isinstance(new_data, pd.DataFrame):
            features = new_data.values
        else:
            features = new_data
        
        # 添加新数据到缓冲区
        self.data_buffer.append(features)
        
        # 如果缓冲区长度超过seq_len，移除最早的数据
        if len(self.data_buffer) > self.seq_len:
            self.data_buffer.pop(0)
    
    def online_predict(self):
        """使用当前缓冲区数据进行在线预测"""
        self.model.eval()
        
        if len(self.data_buffer) < self.seq_len:
            # 如果缓冲区数据不足，返回None或默认值
            return None
        
        # 转换缓冲区数据为numpy数组
        buffer_data = np.array(self.data_buffer)
        
        # 标准化特征
        features_scaled = self.scaler.transform(buffer_data)
        
        # 转换为tensor
        X_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            output = self.model(X_tensor)
        
        return output.item()
    
    def online_update(self, true_value):
        """使用当前缓冲区数据和真实值进行在线模型更新"""
        if len(self.data_buffer) < self.seq_len:
            # 如果缓冲区数据不足，无法更新
            return None
        
        self.model.train()
        
        # 转换缓冲区数据为numpy数组
        buffer_data = np.array(self.data_buffer)
        
        # 标准化特征
        features_scaled = self.scaler.transform(buffer_data)
        
        # 转换为tensor
        X_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor([[true_value]], dtype=torch.float32)
        
        # 前向传播
        output = self.model(X_tensor)
        loss = self.criterion(output, y_tensor)
        
        # 反向传播
        self.online_optimizer.zero_grad()
        loss.backward()
        self.online_optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(path, 'model_weights.pth'))
        # 保存scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
    
    def load_model(self, path):
        # 加载模型权重
        self.model.load_state_dict(torch.load(os.path.join(path, 'model_weights.pth')))
        # 加载scaler
        self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))

def main():
    # 示例用法
    # 注意：实际使用时，需要替换为真实的数据集路径
    print("Attention-LSTM 油泵电机温度预测模型")
    print("请在实际使用时，将数据集路径传递给模型")
    print("模型支持增量学习，可以在部署后根据新数据继续训练")

if __name__ == "__main__":
    main()