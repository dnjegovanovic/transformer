import copy
import torch.nn as nn

from Attention.MultiHeadAttention import MultiHeadAttention
from Attention.PositionalEncoding import PositionalEncoding
from Attention.PositionwiseFeedForward import PositionwiseFeedForward
from Attention.Embeddings import Embeddings
from model.EncoderDecoder import EncoderDecoder
from model.Generator import Generator
from Encoder.Encoder import Encoder
from Encoder.EncoderLayer import EncoderLayer
from Decoder.Decoder import Decoder
from Decoder.DecoderLayer import DecoderLayer


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, number_of_heads,
                 model_dim, position_wise_dim_ff, number_of_layers, dropout=0.1):
        super(Transformer, self).__init__()

        c = copy.deepcopy

        self.mh_attn = MultiHeadAttention(number_of_heads, model_dim)
        self.pwn = PositionwiseFeedForward(model_dim, position_wise_dim_ff, dropout)
        self.position_enc = PositionalEncoding(model_dim, dropout)

        self.encoder = Encoder(EncoderLayer(model_dim, self.mh_attn, c(self.pwn), dropout), number_of_layers)
        self.decoder = Decoder(DecoderLayer(model_dim, self.mh_attn, c(self.pwn), dropout), number_of_layers)

        self.src_pos_emb = nn.Sequential(Embeddings(model_dim, src_vocab_size), c(self.position_enc))
        self.trg_pos_emb = nn.Sequential(Embeddings(model_dim, trg_vocab_size), c(self.position_enc))
        self.generator = Generator(model_dim, trg_vocab_size)
        self.model = EncoderDecoder(self.encoder, self.decoder, self.src_pos_emb, self.trg_pos_emb, self.generator)

    def init_model_params(self):

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def make_model(self):
        self.init_model_params()
        return self.model