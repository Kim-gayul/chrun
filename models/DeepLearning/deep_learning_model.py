import torch.nn as nn

# deeplearning 모델 정의
class DeepLearnModel(nn.Module):
    def __init__(self, input_dim, drop_rate=0.2):
        super().__init__()
        self.lr1 = nn.Linear(input_dim,7)
        self.lr2 = nn.Linear(7,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # 입력데이터가 float64 일 경우 float32로 변환
        x = x.float() 
        out = self.lr1(x)
        out = self.relu(out)
        out = self.dropout(out)
        output = self.lr2(out)

        return output
    

   
