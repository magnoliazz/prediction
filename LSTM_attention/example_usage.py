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
    X, y = predictor.preprocess_data(data, target_col, seq_len=10)
    
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
    new_predictor = OilPumpTemperaturePredictor(input_size=input_size)
    new_predictor.load_model(model_path)
    
    # 测试预测
    print("测试预测...")
    test_data = data.iloc[-10:].drop(columns=[target_col])
    prediction = new_predictor.predict(test_data, seq_len=10)
    print(f"预测的油泵控制器温度: {prediction:.2f}")
    
    # 模拟增量学习
    print("模拟增量学习...")
    # 创建新的示例数据
    new_data = create_sample_data()
    # 取最后20条数据进行增量学习
    incremental_data = new_data.iloc[-20:]
    # 进行增量学习
    new_predictor.incremental_learn(incremental_data, target_col, seq_len=10, epochs=20, batch_size=16)
    
    # 再次测试预测
    print("增量学习后测试预测...")
    test_data = new_data.iloc[-10:].drop(columns=[target_col])
    prediction = new_predictor.predict(test_data, seq_len=10)
    print(f"增量学习后预测的油泵控制器温度: {prediction:.2f}")
    
    print("示例运行完成！")

if __name__ == "__main__":
    main()