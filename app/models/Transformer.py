import torch
import torch.nn as nn

from app.models.InputEmbeddings import InputEmbeddings
from app.models.Encoder import Encoder
from app.models.Decoder import Decoder
from app.models.PositionalEncoding import PositionalEncoding
from app.models.ProjectionLayer import ProjectionLayer


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        proj_layer: ProjectionLayer,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, x, mask):
        x = self.src_embed(x)
        x = self.src_pos(x)
        return self.encoder(x, mask)

    def decode(self, encoder_output, src_mask, target, tgt_mask):
        x = self.tgt_embed(target)
        x = self.tgt_pos(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.proj_layer(x)
