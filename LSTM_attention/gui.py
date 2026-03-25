import tkinter as tk
from tkinter import ttk
from train import OilPumpTemperaturePredictor
import numpy as np

class ModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("油泵电机温度预测模型")
        self.root.geometry("800x600")
        
        # 加载模型
        self.input_size = 23
        self.predictor = OilPumpTemperaturePredictor(input_size=self.input_size)
        self.predictor.load_model('preprocessed_model')
        self.seq_len = 50
        self.predictor.init_buffer(seq_len=self.seq_len)
        
        # 填充初始缓冲区（使用随机数据）
        for i in range(self.seq_len):
            features = np.random.randn(self.input_size)
            self.predictor.update_buffer(features)
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建输入框架
        self.input_frame = ttk.LabelFrame(self.main_frame, text="输入特征", padding="10")
        self.input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建输入字段
        self.input_entries = []
        self.create_input_fields()
        
        # 创建预测按钮
        self.predict_button = ttk.Button(self.main_frame, text="预测", command=self.predict)
        self.predict_button.pack(pady=5)
        
        # 创建预测结果标签
        self.prediction_var = tk.StringVar()
        self.prediction_var.set("预测结果: ")
        self.prediction_label = ttk.Label(self.main_frame, textvariable=self.prediction_var, font=("Arial", 12, "bold"))
        self.prediction_label.pack(pady=5)
        
        # 创建真实值输入框架
        self.true_value_frame = ttk.LabelFrame(self.main_frame, text="真实值", padding="10")
        self.true_value_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.true_value_frame, text="真实温度:").pack(side=tk.LEFT, padx=5)
        self.true_value_entry = ttk.Entry(self.true_value_frame, width=10)
        self.true_value_entry.pack(side=tk.LEFT, padx=5)
        
        # 创建误差计算按钮
        self.calculate_error_button = ttk.Button(self.true_value_frame, text="计算误差", command=self.calculate_error)
        self.calculate_error_button.pack(side=tk.LEFT, padx=5)
        
        # 创建误差结果标签
        self.error_var = tk.StringVar()
        self.error_var.set("误差: ")
        self.error_label = ttk.Label(self.main_frame, textvariable=self.error_var, font=("Arial", 12))
        self.error_label.pack(pady=5)
        
        self.error_percent_var = tk.StringVar()
        self.error_percent_var.set("误差百分比: ")
        self.error_percent_label = ttk.Label(self.main_frame, textvariable=self.error_percent_var, font=("Arial", 12))
        self.error_percent_label.pack(pady=5)
    
    def create_input_fields(self):
        """创建23个输入字段"""
        # 特征名称列表
        feature_names = [
            "直流电压(V)", "直流电流（A）", "1#加载转速(r/min)", "2#加载转速(r/min)",
            "1#加载扭矩(Nm)", "2#加载扭矩(Nm)", "主压（Mpa）", "离合器压力（Mpa）",
            "润滑油压（Mpa）", "制动器压力（Mpa）", "出油温度（℃）", "左电机转速(r/min)",
            "左电机扭矩(Nm)", "右电机转速(r/min)", "右电机扭矩(Nm)", "MCU左侧温度（℃）",
            "MCU右侧温度℃", "左绕组温度（℃）", "右绕组温度（℃）", "右侧轴承温度（℃）",
            "左侧轴承温度（℃）", "油泵转速(r/min)", "油泵控制器温度（℃）"
        ]
        
        # 创建一个网格布局
        for i in range(23):
            row = i // 3
            col = i % 3
            
            label = ttk.Label(self.input_frame, text=feature_names[i]+" ")
            label.grid(row=row, column=col*2, padx=5, pady=5, sticky=tk.W)
            
            entry = ttk.Entry(self.input_frame, width=15)
            entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            self.input_entries.append(entry)
    
    def get_input_values(self):
        """获取输入值"""
        values = []
        for entry in self.input_entries:
            try:
                value = float(entry.get())
            except ValueError:
                value = 0.0
            values.append(value)
        return np.array(values)
    
    def predict(self):
        """进行预测"""
        # 获取输入值
        features = self.get_input_values()
        
        # 更新缓冲区
        self.predictor.update_buffer(features)
        
        # 预测
        prediction = self.predictor.online_predict()
        
        # 显示结果
        self.prediction_var.set(f"预测油泵电机温度: {prediction:.2f} °C")
    
    def calculate_error(self):
        """计算误差"""
        # 获取预测值
        prediction_text = self.prediction_var.get()
        if "预测结果: " in prediction_text:
            prediction = float(prediction_text.split(" ")[2])
        else:
            self.error_var.set("请先进行预测")
            self.error_percent_var.set("")
            return
        
        # 获取真实值
        try:
            true_value = float(self.true_value_entry.get())
        except ValueError:
            self.error_var.set("请输入有效的真实值")
            self.error_percent_var.set("")
            return
        
        # 计算误差
        error = abs(prediction - true_value)
        error_percent = (error / true_value * 100) if true_value != 0 else 0
        
        # 显示结果
        self.error_var.set(f"误差: {error:.2f} °C")
        self.error_percent_var.set(f"误差百分比: {error_percent:.2f}%")
        
        # 进行模型更新
        loss = self.predictor.online_update(true_value)
        print(f"模型更新损失: {loss:.4f}")

def main():
    root = tk.Tk()
    app = ModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()