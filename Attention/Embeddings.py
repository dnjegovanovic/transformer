import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lu_table = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lu_table(x) * math.sqrt(self.d_model)
