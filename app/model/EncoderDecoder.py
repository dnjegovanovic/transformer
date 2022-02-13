import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    Encoder - Decoder architecture like many others.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        :param encoder:
        :param decoder:
        :param src_embed:
        :param tft_embed:
        :param generator:
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Decode input vec macked src and target sequences.
        :param src:
        :param tgt:
        :param src_mask:
        :param tgt_mask:
        :return:
        """

        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
