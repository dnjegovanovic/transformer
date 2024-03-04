import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam

from app.models.InputEmbeddings import InputEmbeddings
from app.models.PositionalEncoding import PositionalEncoding
from app.models.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from app.models.PositionWiseFFN import PositionWiseFFN
from app.models.Encoder import Encoder, EncoderBlock
from app.models.Decoder import Decoder, DecoderBlock
from app.models.ProjectionLayer import ProjectionLayer
from app.models.Transformer import Transformer
from app.utils.get_tokenizer import get_ds
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.__dict__.update(kwargs=kwargs)

        self.src_vocab_size = self.TR_model["src_vocab_size"]
        self.tgt_vocab_size = self.TR_model["tgt_vocab_size"]
        self.src_seq_len = self.TR_model["src_seq_len"]
        self.tgt_seq_len = self.TR_model["tgt_seq_len"]
        self.d_model = self.TR_model["d_model"]
        self.num_layer = self.TR_model["num_layer"]
        self.num_neads = self.TR_model["num_neads"]
        self.dropout = self.TR_model["dropout"]
        self.d_ff = self.TR_model["d_ff"]

        self.config = self.TR_model
        
        self._build_dataset()
        self._build_model()
        self.save_hyperparameters()

    def _build_model(
        self,
    ):
        """_summary_

        Args:
            src_vocab_size (int): size of source vocabulary
            tgt_vocab_size (int): size of target vocabulary
            src_seq_len (int): size of source dequance lenght
            tgt_seq_len (int): size of target dequance lenght
            d_model (int, optional): Model size. Defaults by paper 512.
            num_layer (int, optional): NUmber of layers in encoder/decoder blocks. Defaults to 6.
            num_neads (int, optional): Paper reference. Defaults to 8.
            dropout (float, optional): _description_. Defaults to 0.1.
            d_ff (int, optional): Num of Headen layer of feed forward layer. Defaults to 2.
        """

        # create the embeding layers
        self.src_embeded = InputEmbeddings(self.d_model, self.src_vocab_size)
        self.tgt_embeded = InputEmbeddings(self.d_model, self.tgt_vocab_size)

        # crate the positional encoding layers
        # Can be used only one Positional encoding for encoder and decoder
        # but for process visibility we separate this encoding part
        self.src_pos_enc = PositionalEncoding(
            self.d_model, self.src_seq_len, self.dropout
        )
        self.tgt_pos_enc = PositionalEncoding(
            self.d_model, self.tgt_seq_len, self.dropout
        )

        # create the encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for _ in range(self.num_layer):
            encoder_self_attn = MultiHeadAttentionBlock(
                self.d_model, self.num_neads, self.dropout
            )
            feed_forward_block = PositionWiseFFN(self.d_model, self.d_ff, self.dropout)
            encoder_block = EncoderBlock(
                encoder_self_attn, feed_forward_block, self.dropout
            )

            self.encoder_blocks.append(encoder_block)

        # create the decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for _ in range(self.num_layer):
            decoder_self_attn_block = MultiHeadAttentionBlock(
                self.d_model, self.num_neads, self.dropout
            )
            decoder_cross_attn_block = MultiHeadAttentionBlock(
                self.d_model, self.num_neads, self.dropout
            )
            feed_forward_block = PositionWiseFFN(self.d_model, self.d_ff, self.dropout)
            decoder_block = DecoderBlock(
                decoder_self_attn_block,
                decoder_cross_attn_block,
                feed_forward_block,
                self.dropout,
            )

            self.decoder_blocks.append(decoder_block)

        # Create encoder and decoder
        self.encoder = Encoder(self.encoder_blocks)
        self.decoder = Decoder(self.decoder_blocks)

        # create the projection layer
        self.proj_layer = ProjectionLayer(self.d_model, self.tgt_vocab_size)

        # Crete the Transformer
        self.transformer_model = Transformer(
            self.encoder,
            self.decoder,
            self.src_embeded,
            self.tgt_embeded,
            self.src_pos_enc,
            self.tgt_pos_enc,
            self.proj_layer,
        ).to(device)

        self.model_params = list(self.transformer_model.get_parameter())
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('PAD'), label_smoothing=0.1).to(device)

    def _build_dataset(self):
        (
            self.train_ds,
            self.val_ds,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.max_len_src,
            self.max_len_tgt,
        ) = get_ds(self.config)

    def forward(self, x):
        encoder_input = x['encoder_input'] # (batch, seq_len)
        decoder_input = x['decoder_input'] # (batch, seq_len)
        encoder_mask = x["encoder_mask"] # (batch, 1 ,1 seq_len)
        decoder_mask = x["decoder_mask"] # (batch, 1, seq_len, seq_len)
        
        encoder_out = self.transformer_model.encode(encoder_input, encoder_mask) #(batch, seq_len,d_model)
        decoder_out = self.transformer_model.decode(encoder_out,encoder_mask, decoder_input, decoder_mask) #(batch, seq_len,d_model)
        
        proj_out = self.transformer_model.project(decoder_out) # (batch, seq_len, tgt_voacb_size)
        
        # get label from batch
        label = x['label'] #(batch, seq_len)
        loss = self.loss_fn(proj_out.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        return loss
        
        
    def training_step(self, sample, batch_idx):
        
        encoder_input = sample['encoder_input'] # (batch, seq_len)
        decoder_input = sample['decoder_input'] # (batch, seq_len)
        encoder_mask = sample["encoder_mask"] # (batch, 1 ,1 seq_len)
        decoder_mask = sample["decoder_mask"] # (batch, 1, seq_len, seq_len)
        
        encoder_out = self.transformer_model.encode(encoder_input, encoder_mask) #(batch, seq_len,d_model)
        decoder_out = self.transformer_model.decode(encoder_out,encoder_mask, decoder_input, decoder_mask) #(batch, seq_len,d_model)
        
        proj_out = self.transformer_model.project(decoder_out) # (batch, seq_len, tgt_voacb_size)
        
        # get label from batch
        label = sample['label'] #(batch, seq_len)
        loss = self.loss_fn(proj_out.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        self.log("train_loss", loss.item(), prog_bar=True)
        
        return loss

    def validation_step(self, sample, batch_idx):
        
        encoder_input = sample['encoder_input'] # (batch, seq_len)
        decoder_input = sample['decoder_input'] # (batch, seq_len)
        encoder_mask = sample["encoder_mask"] # (batch, 1 ,1 seq_len)
        decoder_mask = sample["decoder_mask"] # (batch, 1, seq_len, seq_len)
        
        encoder_out = self.transformer_model.encode(encoder_input, encoder_mask) #(batch, seq_len,d_model)
        decoder_out = self.transformer_model.decode(encoder_out,encoder_mask, decoder_input, decoder_mask) #(batch, seq_len,d_model)
        
        proj_out = self.transformer_model.project(decoder_out) # (batch, seq_len, tgt_voacb_size)
        
        # get label from batch
        label = sample['label'] #(batch, seq_len)
        loss = self.loss_fn(proj_out.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        self.log("val_loss", loss.item(), prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        opt_model = Adam([{"params": self.model_params}], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt_model, lr_lambda=lambda epoch: max(0.2, 0.98**self.num_epochs)
        )
        return [opt_model], {"scheduler": scheduler}

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            timeout=30,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            timeout=30,
        )
