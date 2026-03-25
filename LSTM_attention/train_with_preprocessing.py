import pandas as pd
import numpy as np
from train import OilPumpTemperaturePredictor
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.mean = None
        self.std = None
    
    def load_multiple_files(self, file_paths):
        """加载多个Excel文件并合并"""
        data_frames = []
        for path in file_paths:
            df = pd.read_excel(path, header=None)
            data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)
    
    def clean_data(self, data):
        """数据清洗：处理异常值和缺失值"""
        # 处理缺失值
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 处理异常值（使用3σ法则）
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                mean = data[col].mean()
                std = data[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                # 用上下界替换异常值
                data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        return data
    
    def z_score_normalize(self, data, fit=True):
        """Z-score标准化"""
        if fit:
            self.mean = data.mean()
            self.std = data.std()
            self.scaler = {'mean': self.mean, 'std': self.std}
        
        # 避免除零错误
        std = self.std.replace(0, 1)
        normalized_data = (data - self.mean) / std
        
        return normalized_data
    
    def sliding_window(self, data, target_col, seq_len):
        """滑动窗口重构时序数据"""
        # 提取特征和目标
        features = data.drop(columns=[target_col]).values
        target = data[target_col].values
        
        # 构建序列数据
        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(target[i+seq_len])
        
        return np.array(X), np.array(y)
    
    def save_scaler(self, path):
        """保存标准化参数"""
        if self.scaler:
            np.savez(os.path.join(path, 'scaler_params.npz'), mean=self.mean.values, std=self.std.values)
    
    def load_scaler(self, path):
        """加载标准化参数"""
        params = np.load(os.path.join(path, 'scaler_params.npz'))
        self.mean = pd.Series(params['mean'])
        self.std = pd.Series(params['std'])
        self.scaler = {'mean': self.mean, 'std': self.std}

def main():
    # 数据集路径
    train_files = [
        r"H:\试验数据\8数据表1 - 副本.xlsx",
        r"H:\试验数据\8数据表2 - 副本.xlsx",
        r"H:\试验数据\8数据表39 - 副本.xlsx"
    ]
    test_file = r"H:\试验数据\8数据表42 - 副本.xlsx"
    
    # 初始化数据预处理器
    preprocessor = DataPreprocessor()
    
    # 加载并合并训练数据
    print("加载训练数据...")
    train_data = preprocessor.load_multiple_files(train_files)
    print(f"训练数据形状: {train_data.shape}")
    
    # 数据清洗
    print("清洗训练数据...")
    train_data_clean = preprocessor.clean_data(train_data)
    
    # 定义目标列（最后一列：油泵电机温度）
    target_col = train_data_clean.columns[-1]
    print(f"目标列: {target_col}")
    
    # 特征与标签缩放
    print("标准化训练数据...")
    train_features = train_data_clean.drop(columns=[target_col])
    train_target = train_data_clean[target_col]
    
    # 特征归一化
    features_norm_array = preprocessor.z_score_normalize(train_features, fit=True)
    train_features_normalized = pd.DataFrame(features_norm_array, columns=train_features.columns)
    
    # 【修复重点】：目标列（标签）也必须归一化
    target_mean = train_target.mean()
    target_std = train_target.std()
    target_norm_array = (train_target.values - target_mean) / target_std
    
    # 将归一化后的目标列拼凑回去，形成全量归一化数据集
    train_data_normalized = train_features_normalized.copy()
    train_data_normalized[target_col] = target_norm_array
    
    # 重构时序数据：现在传入的是完全归一化后的数据
    print("重构时序数据...")
    seq_len = 50
    X_train, y_train = preprocessor.sliding_window(train_data_normalized, target_col, seq_len)
    print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    
    # 初始化模型
    print("\n初始化模型...")
    input_size = X_train.shape[2]
    predictor = OilPumpTemperaturePredictor(input_size=input_size)
    
    # 确保predictor的scaler也被正确拟合
    # 使用与preprocessor相同的均值和标准差
    predictor.scaler.mean_ = preprocessor.mean.values
    predictor.scaler.scale_ = preprocessor.std.values
    predictor.scaler.n_samples_seen_ = len(train_data_clean)
    predictor.scaler.feature_names_in_ = train_features.columns.tolist()
    
    # 设置目标标签标准化参数
    predictor.target_mean = target_mean
    predictor.target_std = target_std
    
    # 训练模型
    print("训练模型...")
    predictor.train(X_train, y_train, epochs=200, batch_size=32)
    
    # 保存模型和预处理参数
    print("\n保存模型和预处理参数...")
    model_path = 'preprocessed_model'
    os.makedirs(model_path, exist_ok=True)
    predictor.save_model(model_path)
    preprocessor.save_scaler(model_path)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_data = pd.read_excel(test_file, header=None)
    print(f"测试数据形状: {test_data.shape}")
    
    # 获取训练数据的特征列（排除目标列）
    train_feature_cols = train_data_clean.columns[:-1]
    print(f"训练数据特征列数量: {len(train_feature_cols)}")
    
    # 确保测试数据包含相同的特征列
    common_cols = [col for col in train_feature_cols if col in test_data.columns]
    print(f"测试数据与训练数据共有的特征列数量: {len(common_cols)}")
    
    # 如果没有共同特征，使用测试数据的所有列（除了目标列）
    if len(common_cols) == 0:
        print("警告：测试数据与训练数据没有共同特征列，使用测试数据的所有特征")
        # 确保使用整数索引
        test_feature_cols = list(range(len(test_data.columns) - 1))
    else:
        test_feature_cols = common_cols
    
    # 清洗测试数据
    print("清洗测试数据...")
    test_data_clean = preprocessor.clean_data(test_data)
    
    # 使用测试数据自己的最后一列作为目标列
    test_target_col = test_data_clean.columns[-1]
    print(f"测试数据目标列: {test_target_col}")
    
    # === 修复重点区域开始 ===
    
    # 重构测试时序数据（只使用共同特征）
    print("重构测试时序数据...")
    
    # 提取测试集的特征和目标
    test_features = test_data_clean[test_feature_cols]
    test_target = test_data_clean[test_target_col]
    
    # 1. 对测试集特征进行标准化 (必须使用 preprocessor 中保存的训练集均值和标准差)
    # 使用 replace(0, 1) 防止除以 0
    test_features_normalized = (test_features - preprocessor.mean[test_feature_cols]) / preprocessor.std[test_feature_cols].replace(0, 1)
    
    # 2. 重新拼接：标准化的特征 + 原始的真实温度目标列
    test_data_normalized = test_features_normalized.copy()
    test_data_normalized[test_target_col] = test_target.values
    
    # 3. 滑动窗口切分
    X_test, y_test = preprocessor.sliding_window(test_data_normalized, test_target_col, seq_len)
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    
    # 评估模型
    print("\n评估模型...")
    test_input_size = X_test.shape[2]
    if test_input_size != input_size:
        print(f"警告：测试数据特征数量 ({test_input_size}) 与模型输入大小 ({input_size}) 不匹配")
    else:
        predictor.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1) # 这里是真实的物理温度
            
            # 模型输出的是标准化的预测值
            outputs = predictor.model(X_test_tensor)
            
            # 4. 反归一化：将预测值还原到物理尺度
            outputs_real = outputs * target_std + target_mean
            
            # 5. 真实值无需处理，直接用于比较
            y_test_real = y_test_tensor
            
            # 计算真实尺度下的误差
            mse = torch.mean((outputs_real - y_test_real) ** 2).item()
            mae = torch.mean(torch.abs(outputs_real - y_test_real)).item()
            rmse = torch.sqrt(torch.mean((outputs_real - y_test_real) ** 2)).item()
        
        print(f"测试集评估结果:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
    
    print("\n模型训练和评估完成！")

if __name__ == "__main__":
    import torch
    main()