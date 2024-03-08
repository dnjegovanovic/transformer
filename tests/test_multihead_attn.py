import torch

from transformer_app.models import MultiHeadAttentionBlock as MHAB


def test_multiheadattention_block():
    h = 8
    q = torch.rand(2, 6, 512)  # (batch, seq_len, emb_size)
    k = torch.rand(2, 6, 512)  # (batch, seq_len, emb_size)
    v = torch.rand(2, 6, 512)  # (batch, seq_len, emb_size)

    mha = MHAB.MultiHeadAttentionBlock(512, h, 0.1)

    output = mha.forward(q, k, v)  # (batch, seq_len, emb_size)
    print("Output shape: {}".format(output.shape))

    assert output.shape == q.shape
