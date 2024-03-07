import torch
import torch.nn as nn
import math

from models.MultiHeadAttentionBlock import MultiHeadAttentionBlock as MHAB
from models.MultiHeadAttentionBlock import ResidualConnection as ResCon
from models.PositionWiseFFN import PositionWiseFFN as PFFN
from models.LayerNormalization import LayerNormalization as LN


class EncoderBlock(nn.Module):
    def __init__(
        self,self_attn_block: MHAB, feed_forward_net: PFFN, dropout: float
    ) -> None:
        super().__init__()

        self.self_attn_block = self_attn_block
        self.feed_forward_net = feed_forward_net
        self.res_connection = nn.ModuleList([ResCon(dropout) for _ in range(2)])

    def forward(self, x, mask=None):
        x = self.res_connection[0](x, lambda x: self.self_attn_block(x, x, x, mask))
        x = self.res_connection[1](x, self.feed_forward_net)

        return x


class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm_layer = LN()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm_layer(x)
