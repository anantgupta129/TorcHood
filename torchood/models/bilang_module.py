from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import BLEUScore, CharErrorRate, WordErrorRate

from torchood.data.components.opus_books import casual_mask

from .components.bilang_transformer import build_transformer


class BiLangLitModule(LightningModule):
    def __init__(
        self,
        learning_rate: float,
        config: dict,
        tokenizer_src: Any,
        tokenizer_tgt: Any,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.seq_len = config["seq_len"]
        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

        self.net = build_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.seq_len,
            self.seq_len,
            d_model=config["d_model"],
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAF]"), label_smoothing=0.1
        )

        self.train_loss = []
        # for validation
        self.source_texts = []
        self.expected = []
        self.predicted = []
        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()
        self.bleu_score = BLEUScore()

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        # run the tensors through the encoder, decider & projection layer
        encoder_output = self.net.encode(encoder_input, encoder_mask)  # (B, seq_len, seq_len)
        decoder_output = self.net.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.net.project(decoder_output)  # (B, seq_len, vocab_size)

        return proj_output

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        proj_output = self.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)

        label = batch["label"]  # (B, seq_len)
        loss = self.criterion(proj_output.view(-1, self.tgt_vocab_size), label.view(-1))

        # update and log metrics
        self.train_loss.append(loss)
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.log("train/loss", mean_loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss = []

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]
        # check if batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        out = self.greedy_decode(
            encoder_input, encoder_mask, max_len=self.seq_len, device=self.device
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        out_text = self.tokenizer_tgt.decode(out.detach().cpu().numpy())

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(out_text)

    def on_validation_epoch_end(self) -> None:
        cer = self.char_error_rate(self.predicted, self.expected)
        wer = self.word_error_rate(self.predicted, self.expected)
        bleu = self.bleu_score(self.predicted, self.expected)

        self.source_texts = []
        self.expected = []
        self.predicted = []

        self.log("val/cer", cer, prog_bar=True)
        self.log("val/wer", wer, prog_bar=True)
        self.log("val/bleu", bleu, prog_bar=True)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-9)
        # num_epochs = self.trainer.max_epochs
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.learning_rate,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     epochs=num_epochs,
        #     pct_start=5 / num_epochs,
        #     div_factor=100,
        #     three_phase=False,
        #     final_div_factor=100,
        #     anneal_strategy="linear",
        # )
        # lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer]

    def greedy_decode(self, source, source_mask, max_len: int, device: str):
        sos_idx = self.tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = self.tokenizer_tgt.token_to_id("[EOS]")

        # encoder output
        encoder_output = self.net.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break
            # build target mask
            decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = self.net.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = self.net.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
                ],
                dim=1,
            )
            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)
