import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    def __init__(self,features: int=1, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.zeros(2))  # added

    def forward(self, x: torch.Tensor):
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
