import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.InputEmbeddings import InputEmbeddings
from models.PositionalEncoding import PositionalEncoding
from models.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from models.PositionWiseFFN import PositionWiseFFN
from models.Encoder import Encoder, EncoderBlock
from models.Decoder import Decoder, DecoderBlock
from models.ProjectionLayer import ProjectionLayer
from models.Transformer import Transformer
from utils.get_tokenizer import get_ds
from dataset.BilingualDataSet import casual_mask
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.__dict__.update(kwargs)

        self.src_vocab_size = self.TR_model["src_vocab_size"]
        self.tgt_vocab_size = self.TR_model["tgt_vocab_size"]
        self.src_seq_len = self.TR_model["src_seq_len"]
        self.tgt_seq_len = self.TR_model["tgt_seq_len"]
        self.d_model = self.TR_model["d_model"]
        self.num_layer = self.TR_model["num_layer"]
        self.num_neads = self.TR_model["num_neads"]
        self.dropout = self.TR_model["dropout"]
        self.d_ff = self.TR_model["d_ff"]
        self.max_len = self.TR_model["seq_len"]
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

        self.model_params = list(self.transformer_model.parameters())
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1
        ).to(device)

    def _build_dataset(self):
        (
            self.train_ds,
            self.val_ds,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.max_len_src,
            self.max_len_tgt,
        ) = get_ds(self.config)
        
        self.src_vocab_size = self.tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()
        
    def test_on_validation(self, num_of_batch: int = 1):
        self.transformer_model.eval()
        
        couter = 0
        console_width = 80
        with torch.no_grad():
            val_ds = self.val_dataloader()
            for batch in val_ds:
                if couter > num_of_batch:
                    print('-'*console_width)
                    break
                encoder_input = batch["encoder_input"]  # (batch, seq_len)
                encoder_mask = batch["encoder_mask"]  # (batch, 1 ,1 seq_len)
                
                            # check that the batch size is 1
                assert encoder_input.size(
                    0) == self.TR_model["batch_size"], "Batch size must be 1 for validation"
                
                c_batch = 0
                for e_i, e_m in  zip(encoder_input, encoder_mask):
                    model_out = self._greedy_decode(e_i.to(device), e_m.to(device), self.tokenizer_src, self.tokenizer_tgt, self.max_len, device)
                    source_text = batch["src_text"][c_batch]
                    target_text = batch["tgt_text"][c_batch]
                    model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
                    
                    couter += 1
                    
                    # Print the source, target and model output
                    print('-'*console_width)
                    print(f"{f'SOURCE: ':>12}{source_text}")
                    print(f"{f'TARGET: ':>12}{target_text}")
                    print(f"{f'PREDICTED: ':>12}{model_out_text}")
                    c_batch += 1

                
    
    def _greedy_decode(self,source, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = self.transformer_model.encode(source, src_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = casual_mask(decoder_input.size(1)).type_as(src_mask).to(device)

            # calculate output
            out = self.transformer_model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

            # get next token
            prob = self.transformer_model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)
    
    def forward(self, x):
        encoder_input = x["encoder_input"]  # (batch, seq_len)
        decoder_input = x["decoder_input"]  # (batch, seq_len)
        encoder_mask = x["encoder_mask"]  # (batch, 1 ,1 seq_len)
        decoder_mask = x["decoder_mask"]  # (batch, 1, seq_len, seq_len)

        encoder_out = self.transformer_model.encode(
            encoder_input, encoder_mask
        )  # (batch, seq_len,d_model)
        decoder_out = self.transformer_model.decode(
            encoder_out, encoder_mask, decoder_input, decoder_mask
        )  # (batch, seq_len,d_model)

        proj_out = self.transformer_model.project(
            decoder_out
        )  # (batch, seq_len, tgt_voacb_size)

        return proj_out

    def training_step(self, sample, batch_idx):
        proj_out = self(sample)

        # get label from batch
        label = sample["label"]  # (batch, seq_len)
        loss = self.loss_fn(
            proj_out.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )

        self.log("train_loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, sample, batch_idx):
        
        proj_out = self(sample)
        # get label from batch
        label = sample["label"]  # (batch, seq_len)
        loss = self.loss_fn(
            proj_out.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )

        self.log("val_loss", loss.item(), prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt_model = Adam([{"params": self.model_params}], lr=self.TR_model["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt_model, lr_lambda=lambda epoch: max(0.2, 0.98**self.TR_model["epochs"])
        )
        return [opt_model], {"scheduler": scheduler}

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.TR_model["batch_size"],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            timeout=120,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.TR_model["batch_size"],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            timeout=120,
        )
