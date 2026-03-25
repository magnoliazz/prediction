import pandas as pd
import numpy as np
from train import OilPumpTemperaturePredictor
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.mean = None
        self.std = None
    
    def load_scaler(self, path):
        """加载标准化参数"""
        params = np.load(os.path.join(path, 'scaler_params.npz'))
        self.mean = pd.Series(params['mean'])
        self.std = pd.Series(params['std'])
        self.scaler = {'mean': self.mean, 'std': self.std}
        return self.mean.index.tolist()

def main():
    # 数据集路径
    test_file = r"H:\试验数据\8数据表1 - 副本.xlsx"
    
    # 加载预处理器和特征列
    preprocessor = DataPreprocessor()
    model_path = 'preprocessed_model'
    
    try:
        # 加载训练时的特征列
        feature_cols = preprocessor.load_scaler(model_path)
        print(f"训练时的特征列数量: {len(feature_cols)}")
        print(f"前5个特征列: {feature_cols[:5]}")
    except Exception as e:
        print(f"加载特征列失败: {e}")
        return
    
    # 读取测试数据
    print("\n读取测试数据...")
    test_data = pd.read_excel(test_file, header=None)
    print(f"测试数据形状: {test_data.shape}")
    
    # 使用测试数据的前N列作为特征，其中N是模型的输入大小
    input_size = len(feature_cols)
    if input_size > len(test_data.columns) - 1:
        print(f"警告：测试数据的特征列数量不足，需要{input_size}列，实际只有{len(test_data.columns) - 1}列")
        return
    
    # 使用前input_size列作为特征，最后一列作为目标
    test_feature_cols = list(range(input_size))
    target_col = test_data.columns[-1]
    print(f"测试数据目标列: {target_col}")
    
    # 初始化模型
    print(f"\n模型输入大小: {input_size}")
    online_predictor = OilPumpTemperaturePredictor(input_size=input_size)
    
    # 加载模型
    print("加载模型...")
    try:
        online_predictor.load_model(model_path)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 初始化滑动窗口
    print("初始化滑动窗口...")
    seq_len = 50
    online_predictor.init_buffer(seq_len=seq_len)
    
    # 准备测试数据
    test_data_subset = test_data[test_feature_cols + [target_col]]
    test_features = test_data_subset.drop(columns=[target_col])
    test_targets = test_data_subset[target_col].values
    
    print(f"测试数据子集形状: {test_data_subset.shape}")
    
    # 填充初始缓冲区
    print("\n填充初始缓冲区...")
    for i in range(min(seq_len, len(test_features))):
        # 确保数据长度与特征数量匹配
        if len(test_features.iloc[i].values) == input_size:
            online_predictor.update_buffer(test_features.iloc[i].values)
        else:
            print(f"数据长度不匹配: {len(test_features.iloc[i].values)} != {input_size}")
            return
    
    # 开始在线预测和更新
    print("\n开始在线预测和更新...")
    total_loss = 0
    predictions = []
    update_count = 0
    
    for i in range(seq_len, len(test_data_subset)):
        # 1. 在线预测
        prediction = online_predictor.online_predict()
        
        # 2. 获取真实值
        true_value = test_targets[i]
        
        # 3. 在线更新
        loss = online_predictor.online_update(true_value)
        
        # 4. 更新缓冲区，添加新数据
        if len(test_features.iloc[i].values) == input_size:
            online_predictor.update_buffer(test_features.iloc[i].values)
        
        # 记录结果
        if prediction is not None and loss is not None and loss > 0:
            predictions.append(prediction)
            total_loss += loss
            update_count += 1
            
            # 每10步打印一次
            if (i - seq_len + 1) % 10 == 0:
                print(f"Step {i - seq_len + 1}/{len(test_data_subset) - seq_len}, Loss: {loss:.4f}, Prediction: {prediction:.2f}, True: {true_value:.2f}")
    
    # 计算平均损失
    if update_count > 0:
        avg_loss = total_loss / update_count
        print(f"\n平均在线更新损失: {avg_loss:.4f}")
        print(f"更新次数: {update_count}")
        
        if len(predictions) > 0:
            # 计算预测误差
            predictions = np.array(predictions)
            true_values = test_targets[seq_len:seq_len+len(predictions)]
            mse = np.mean((predictions - true_values) ** 2)
            mae = np.mean(np.abs(predictions - true_values))
            
            # 计算偏移量（百分比）
            # 避免除以零
            non_zero_mask = true_values != 0
            if np.any(non_zero_mask):
                offset_percentage = np.mean(np.abs((predictions[non_zero_mask] - true_values[non_zero_mask]) / true_values[non_zero_mask]) * 100)
            else:
                offset_percentage = 0.0
            
            print(f"在线预测MSE: {mse:.4f}")
            print(f"在线预测MAE: {mae:.4f}")
            print(f"平均偏移量: {offset_percentage:.2f}%")
    
    print("\n在线预测和更新测试完成！")

if __name__ == "__main__":
    main()