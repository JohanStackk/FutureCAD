from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x)))
