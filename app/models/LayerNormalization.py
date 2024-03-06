import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    def __init__(self,features: int=1, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(512))  # multiplied
        self.bias = nn.Parameter(torch.zeros(512))  # added

    def forward(self, x: torch.Tensor):
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        t1 = self.alpha * (x - mean)
        t2 = (std + self.eps) + self.bias
        res = t1 / t2
        return res
