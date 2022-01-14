import torch.nn as nn
from Encoder.Encoder import clones
from Encoder.SublayerConnection import SublayerConnection


class EncoderLayer(nn.Module):
    """
        Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple,
        position-wise fully connected feed- forward network.
        Encoder is made up of self attention and feed forward.
    """

    def __init__(self, size, self_attn, feed_forward, dropout, N=2):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.size = size
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), N)

    def forward(self, x, mask):
        for subl in self.sublayer:
            x = subl(x, lambda x: self.self_attn(x, x, x, mask))
        return x
