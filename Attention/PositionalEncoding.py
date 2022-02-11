import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout_p, max_sequence_length=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_p)

        self.position_id = torch.arange(0, max_sequence_length).unsqueeze(1)
        self.freq = torch.exp(torch.arange(0, model_dimension, 2) * -(math.log(10000.0) / model_dimension))

        self.positional_table = torch.zeros(max_sequence_length, model_dimension)
        self.positional_table[:, 0::2] = torch.sin(self.positional_table * self.freq)
        self.positional_table[:, 1::2] = torch.cos(self.positional_table * self.freq)

        self.positional_table = self.positional_table.unsqueeze(0)
 
        self.register_buffer('positional_table', self.positional_table)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.positional_table[:, :x.size(1)],
                                        requires_grad=False)

        return self.dropout(x)
