import pandas as pd
import numpy as np
from train import OilPumpTemperaturePredictor
import os

def create_sample_data():
    # 创建示例数据
    np.random.seed(42)
    data = {
        '直流电压': np.random.uniform(60, 360, 1000),
        '直流电流': np.random.uniform(-650, -150, 1000),
        '1#加载转速': np.random.uniform(-160, 7200, 1000),
        '2#加载转速': np.random.uniform(-160, 7200, 1000),
        '1#加载扭矩': np.random.uniform(1.7, 2.0, 1000),
        '2#加载扭矩': np.random.uniform(1.7, 2.0, 1000),
        '主压': np.random.uniform(0.39, 0.41, 1000),
        '离合器压力': np.random.uniform(-0.02, -0.02, 1000),
        '润滑油压': np.random.uniform(39, 50, 1000),
        '制动器压力': np.random.uniform(500, 2400, 1000),
        '出油温度': np.random.uniform(40, 80, 1000),
        '左电机转速': np.random.uniform(-1000, 1000, 1000),
        '左电机扭矩': np.random.uniform(-400, 700, 1000),
        '右电机转速': np.random.uniform(-1000, 1000, 1000),
        '右电机扭矩': np.random.uniform(-400, 700, 1000),
        'MCU左侧温度': np.random.uniform(40, 80, 1000),
        'MCU右侧温度': np.random.uniform(40, 80, 1000),
        '左绕组温度': np.random.uniform(35, 125, 1000),
        '右绕组温度': np.random.uniform(35, 125, 1000),
        '右侧轴承温度': np.random.uniform(34, 85, 1000),
        '左侧轴承温度': np.random.uniform(35, 85, 1000),
        '油泵转速': np.random.uniform(2200, 2400, 1000),
        '油泵控制器温度': np.random.uniform(34, 75, 1000)
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    # 创建示例数据
    print("创建示例数据...")
    data = create_sample_data()
    
    # 定义目标列
    target_col = '油泵控制器温度'
    
    # 计算输入特征数量
    input_size = len(data.columns) - 1
    
    # 初始化模型
    print("初始化模型...")
    predictor = OilPumpTemperaturePredictor(input_size=input_size)
    
    # 预处理数据
    print("预处理数据...")
    seq_len = 50  # 滑动窗口长度
    X, y = predictor.preprocess_data(data, target_col, seq_len=seq_len)
    
    # 训练模型
    print("训练模型...")
    predictor.train(X, y, epochs=50, batch_size=32)
    
    # 保存模型
    print("保存模型...")
    model_path = 'model'
    os.makedirs(model_path, exist_ok=True)
    predictor.save_model(model_path)
    
    # 加载模型
    print("加载模型...")
    online_predictor = OilPumpTemperaturePredictor(input_size=input_size)
    online_predictor.load_model(model_path)
    
    # 初始化滑动窗口
    print("初始化滑动窗口...")
    online_predictor.init_buffer(seq_len=seq_len)
    
    # 模拟在线预测和更新过程
    print("模拟在线预测和更新过程...")
    
    # 准备测试数据
    test_data = create_sample_data()
    test_features = test_data.drop(columns=[target_col])
    test_targets = test_data[target_col].values
    
    # 填充初始缓冲区
    print("填充初始缓冲区...")
    for i in range(seq_len):
        online_predictor.update_buffer(test_features.iloc[i].values)
    
    # 开始在线预测和更新
    print("开始在线预测和更新...")
    total_loss = 0
    predictions = []
    
    for i in range(seq_len, len(test_data)):
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
            
            # 每100步打印一次
            if (i - seq_len + 1) % 100 == 0:
                print(f"Step {i - seq_len + 1}/{len(test_data) - seq_len}, Loss: {loss:.4f}, Prediction: {prediction:.2f}, True: {true_value:.2f}")
    
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
    
    print("在线预测和更新模拟完成！")

if __name__ == "__main__":
    main()