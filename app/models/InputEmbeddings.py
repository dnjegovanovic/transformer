import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, emb_size: int, vocab_size: int) -> None:
        """_summary_

        Args:
            emb_size (int): size of embedded vector mostly 512
            vocab_size (int): vocabulary size
        """
        super().__init__()

        self.emb_size = emb_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        return self.embedding * math.sqrt(
            self.emb_size
        )  # from the paper multiplay embedded by sqrt of the emb_size(d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.emb_size = emb_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # creting matrix shape of