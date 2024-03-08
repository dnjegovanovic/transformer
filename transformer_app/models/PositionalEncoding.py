import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.emb_size = emb_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # creting matrix shape of (seq_len, emb_size)
        pos_enc = torch.zeros(seq_len, emb_size)

        # create vec of shape (seq_len)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # positional encoding formula
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)
        )  # numerical stabiliti

        # applay sin cos
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        pos_enc = pos_enc.unsqueeze(
            0
        )  # (1, seq_len, emb_size) add one more dim for batch

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        x = x + (self.pos_enc[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
