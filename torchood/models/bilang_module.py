from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LambdaLR

from .components.bilang_transformer import build_transformer


class BiLangLitModule(LightningModule):
    def __init__(self, learning_rate: float, criterion: Any, vocab_size: int) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.criterion = criterion

        self.net = build_transformer()

        self.train_loss = []
        self.val_loss = []

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        # run the tensors through the encoder, decider & projection layer
        encoder_output = self.net.encode(encoder_input, encoder_mask)  # (B, seq_len, seq_len)
        decoder_output = self.net.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.net.project(decoder_output)  # (B, seq_len, vocab_size)

        return [proj_output, decoder_output, encoder_output]

    def _step(self, batch: Any) -> torch.Tensor:
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        proj_output, decoder_output, encoder_output = self.forward(
            encoder_input, decoder_input, encoder_mask, decoder_mask
        )

        label = batch["label"]  # (B, seq_len)
        loss = self.criterion(proj_output.view(-1, self.vocab_size), label.view(-1))

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss = self._step(batch)
        # update and log metrics
        self.train_loss.append(loss)
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.log("train/loss", mean_loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss = []

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss = self._step(batch)
        # update and log metrics
        self.val_loss.append(loss)
        mean_loss = sum(self.val_loss) / len(self.val_loss)
        self.log("val/loss", mean_loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_loss = []

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-9)
        num_epochs = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=num_epochs,
            pct_start=5 / num_epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [lr_scheduler]
