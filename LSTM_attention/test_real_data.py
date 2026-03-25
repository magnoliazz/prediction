import pandas as pd
import numpy as np
from train import OilPumpTemperaturePredictor
import os

def main():
    # 数据集路径
    data_path = r"H:\试验数据\8数据表1 - 副本.xlsx"
    
    # 读取Excel文件
    print("读取数据集...")
    data = pd.read_excel(data_path)
    
    # 定义目标列（最后一列：油泵电机温度）
    target_col = data.columns[-1]
    print(f"目标列: {target_col}")
    
    # 计算输入特征数量
    input_size = len(data.columns) - 1
    print(f"输入特征数量: {input_size}")
    
    # 加载训练好的模型
    print("\n加载模型...")
    model_path = 'real_model'
    online_predictor = OilPumpTemperaturePredictor(input_size=input_size)
    online_predictor.load_model(model_path)
    
    # 初始化滑动窗口
    print("初始化滑动窗口...")
    seq_len = 50  # 滑动窗口长度
    online_predictor.init_buffer(seq_len=seq_len)
    
    # 准备测试数据
    test_features = data.drop(columns=[target_col])
    test_targets = data[target_col].values
    
    # 填充初始缓冲区
    print("填充初始缓冲区...")
    for i in range(seq_len):
        online_predictor.update_buffer(test_features.iloc[i].values)
    
    # 开始在线预测和更新
    print("\n开始在线预测和更新...")
    total_loss = 0
    predictions = []
    
    for i in range(seq_len, len(data)):
        # 1. 在线预测
        prediction = online_predictor.online_predict()
        
        # 2. 获取真实值
        true_value = test_targets[i]
        
        # 3. 在线更新
        loss = online_predictor.online_update(true_value)
        
        # 4. 更新缓冲区，添加新数据
        online_predictor.update_buffer(test_features.iloc[i].values)
        
        # 记录结果
        if prediction is not None and loss is not None:
            predictions.append(prediction)
            total_loss += loss
            
            # 每10步打印一次
            if (i - seq_len + 1) % 10 == 0:
                print(f"Step {i - seq_len + 1}/{len(data) - seq_len}, Loss: {loss:.4f}, Prediction: {prediction:.2f}, True: {true_value:.2f}")
    
    # 计算平均损失
    if len(predictions) > 0:
        avg_loss = total_loss / len(predictions)
        print(f"\n平均在线更新损失: {avg_loss:.4f}")
        
        # 计算预测误差
        predictions = np.array(predictions)
        true_values = test_targets[seq_len:seq_len+len(predictions)]
        mse = np.mean((predictions - true_values) ** 2)
        mae = np.mean(np.abs(predictions - true_values))
        print(f"在线预测MSE: {mse:.4f}")
        print(f"在线预测MAE: {mae:.4f}")
    
    print("\n在线预测和更新测试完成！")

if __name__ == "__main__":
    main()