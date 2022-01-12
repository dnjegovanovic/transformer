import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    Residual connection followed by  a layer norm and dropout.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
