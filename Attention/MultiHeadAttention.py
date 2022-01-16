import torch.nn as nn
from utils.attention import calc_attention
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
