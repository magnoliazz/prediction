import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # 注意力机制层
        self.attention = nn.Linear(hidden_size, 1)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # LSTM前向传播
        out, (hidden, cell) = self.lstm(x)
        # out shape: (batch_size, seq_len, hidden_size)
        
        # 计算注意力权重
        attn_weights = F.softmax(self.attention(out), dim=1)
        # attn_weights shape: (batch_size, seq_len, 1)
        
        # 加权平均
        context = torch.sum(attn_weights * out, dim=1)
        # context shape: (batch_size, hidden_size)
        
        # 输出预测
        output = self.fc(context)
        # output shape: (batch_size, 1)
        
        return output