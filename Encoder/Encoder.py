import torch.nn as nn
import math, copy, time


def clones(module, N):
    """
    The encoder is composed of stack of N = 6 identical layers
    This fun multiply layers.
    :param module:
    :param N:
    :return:
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        # We can use Custom implementation of LayerNorm
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input through each layer.
        :param x:
        :param mask:
        :return:
        """

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
