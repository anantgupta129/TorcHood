from typing import Any, Tuple

import cv2
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .components.unet import DiceLoss, UNet


class UNetLitModule(LightningModule):
    def __init__(
        self,
        learning_rate: float,
        out_channels: int,
        in_channels: int = 3,
        strided_conv: bool = False,
        upsample_mode: str = "transpose",
        criterion_type: str = "dice",
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.net = UNet(in_channels, out_channels, strided_conv, upsample_mode)

        if criterion_type == "dice":
            self.criterion = DiceLoss()
        elif criterion_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown criterion type: {criterion_type} - must be 'dice' or 'ce'")

        self.train_loss = []
        self.val_loss = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: Any) -> Tuple[torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        return loss, logits, y

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, logits, targets = self._step(batch)

        # update and log metrics
        self.train_loss.append(loss)
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.log("train/loss", mean_loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss = []

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, logits, targets = self._step(batch)

        # update and log metrics
        self.val_loss.append(loss)
        mean_loss = sum(self.val_loss) / len(self.val_loss)
        self.log("val/loss", mean_loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_loss = []

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=1e-9)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=self.trainer.max_epochs,
            pct_start=0.3,
            div_factor=10,
            three_phase=True,
            final_div_factor=10,
            anneal_strategy="linear",
        )
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [lr_scheduler]


def add_title(img, title):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    thickness = 2
    scale = 0.5
    cv2.putText(img, title, (10, 30), font, scale, color, thickness, cv2.LINE_AA)
    return img
