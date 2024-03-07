import torch
import torch.nn as nn

from models.MultiHeadAttentionBlock import MultiHeadAttentionBlock as MHAB
from models.MultiHeadAttentionBlock import ResidualConnection as ResCon
from models.PositionWiseFFN import PositionWiseFFN as PFFN
from models.LayerNormalization import LayerNormalization as LN


class DecoderBlock(nn.Module):
    def __init__(
        self, self_attn: MHAB, cross_attn: MHAB, ff_net: PFFN, dropout: float
    ) -> None:
        super().__init__()

        self.self_attn_block = self_attn
        self.cross_attn_block = cross_attn
        self.ffn = ff_net
        self.res_connection = nn.ModuleList(ResCon(dropout) for _ in range(3))

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        """_summary_

        Args:
            x (_type_): input
            ecoder_output (_type_): k,v from encoder
            ecoder_mask (_type_): mask for language 1
            decoder_mask (_type_): mask for language 2
        """

        x = self.res_connection[0](
            x, lambda x: self.self_attn_block(x, x, x, decoder_mask)
        )
        x = self.res_connection[1](
            x,
            lambda x: self.cross_attn_block(
                x, encoder_output, encoder_output, encoder_mask
            ),
        )
        x = self.res_connection[1](x, lambda x: self.ffn(x))

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm_layer = LN()

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)

        return self.norm_layer(x)
