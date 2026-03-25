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
    
    # 打印数据信息
    print(f"数据形状: {data.shape}")
    print("数据列名:")
    for i, col in enumerate(data.columns):
        print(f"{i+1}. {col}")
    
    # 定义目标列（最后一列：油泵电机温度）
    target_col = data.columns[-1]
    print(f"\n目标列: {target_col}")
    
    # 计算输入特征数量
    input_size = len(data.columns) - 1
    print(f"输入特征数量: {input_size}")
    
    # 初始化模型
    print("\n初始化模型...")
    predictor = OilPumpTemperaturePredictor(input_size=input_size)
    
    # 预处理数据
    print("预处理数据...")
    seq_len = 50  # 滑动窗口长度
    X, y = predictor.preprocess_data(data, target_col, seq_len=seq_len)
    
    print(f"预处理后数据形状: X={X.shape}, y={y.shape}")
    
    # 训练模型
    print("\n训练模型...")
    predictor.train(X, y, epochs=100, batch_size=32)
    
    # 保存模型
    print("\n保存模型...")
    model_path = 'real_model'
    os.makedirs(model_path, exist_ok=True)
    predictor.save_model(model_path)
    
    print("\n模型训练完成并保存！")

if __name__ == "__main__":
    main()