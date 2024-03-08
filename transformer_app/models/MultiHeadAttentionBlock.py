import math

import torch
import torch.nn as nn

from .LayerNormalization import LayerNormalization


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, emb_size: int, head_num: int, dropout: float) -> None:
        """_summary_

        Args:
            emb_size (int): _description_
            head_num (int): _description_
            dropout (float): _description_
        """
        super().__init__()

        self.emb_size = emb_size
        self.head_num = head_num

        assert emb_size % head_num == 0

        self.d_k = emb_size // head_num

        # Construct Wk, Wv, Wq
        self.w_q = nn.Linear(emb_size, emb_size)
        self.w_k = nn.Linear(emb_size, emb_size)
        self.w_v = nn.Linear(emb_size, emb_size)
        self.w_o = nn.Linear(emb_size, emb_size)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]

        # q=(batch, head_num, seq_len, d_k), k=(batch, head_num, d_k, seq_len), result --> (batch, head_num, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # where the mask is  0  replaze attn_score value with -1e9
            attn_scores.masked_fill_(mask == 0, -1e9)

        attn_scores = attn_scores.softmax(dim=-1)  # (batch,num_head,seq_len,seq_len)

        if dropout is not None:
            attn_scores = dropout(attn_scores)

        # (attn_scores @ v) --> output for next layer, attn_scores --> for visualization
        return (attn_scores @ v), attn_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # (batch, seq_len, emb_size) --> (batch, seq_len, emb_size)
        key = self.w_k(k)  # (batch, seq_len, emb_size) --> (batch, seq_len, emb_size)
        values = self.w_v(
            v
        )  # (batch, seq_len, emb_size) --> (batch, seq_len, emb_size)

        # (batch, seq_len, emb_size) -->(batch, seq_len, head_num, d_k) -->transpose (batch, head_num, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.head_num, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.head_num, self.d_k).transpose(
            1, 2
        )
        values = values.view(
            values.shape[0], values.shape[1], self.head_num, self.d_k
        ).transpose(1, 2)

        x, self.attn_scores = MultiHeadAttentionBlock.attention(
            query, key, values, mask, self.dropout
        )

        # concanate all head
        # (batch, num_heads, seq_len, d_k) --> (batch, seq_len, num_head, d_k) --> (batch, seq_len, emb_size)
        # self.head_num * self.d_k = self.emb_size
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.head_num * self.d_k)
        )

        # (batch, seq_len, emb_size)  --> (batch, seq_len, emb_size)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.layer_norm(x)))
