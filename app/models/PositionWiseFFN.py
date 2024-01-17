import torch
import torch.nn as nn
import math


class PositionWiseFFN(nn.Module):
    """IMplementation of Position-wised feed forward netwrok

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, dff_model: int, dropout: float) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model, dff_model)  # w1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff_model, d_model)  # w2 and B2

    def forward(self, x: torch.Tensor):
        # x: (bathc, seq_len, d_model) --> (batch, seq_len, dff_model) --> (bathc, seq_len, d_model)

        out = self.linear_1(x)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)

        return out
