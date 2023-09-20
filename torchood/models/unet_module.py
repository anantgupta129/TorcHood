from typing import Any, Tuple

import cv2
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .components.unet import DiceLoss, UNet, mean_iou, pixel_accuracy


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
        self.nc = out_channels
        self.net = UNet(in_channels, out_channels, strided_conv, upsample_mode)

        self.criterion_type = criterion_type
        if criterion_type == "dice":
            self.criterion = DiceLoss()
        elif criterion_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown criterion type: {criterion_type} - must be 'dice' or 'ce'")

        self.train_loss = []
        self.train_pixel_acc = []
        self.train_mIoU = []
        self.val_loss = []
        self.val_pixel_acc = []
        self.val_mIoU = []

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
        self.log(f"train/{self.criterion_type}_loss", mean_loss, prog_bar=True)

        self.train_pixel_acc.append(pixel_accuracy(logits, targets))
        px_acc = sum(self.train_pixel_acc) / len(self.train_pixel_acc)
        self.log("train/pixel_accuracy", px_acc, prog_bar=True)

        self.train_mIoU.append(mean_iou(logits, targets, self.nc))
        miou = sum(self.train_mIoU) / len(self.train_mIoU)
        self.log("train/meanIoU", miou, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss = []
        self.train_pixel_acc = []
        self.train_mIoU = []

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, logits, targets = self._step(batch)

        # update and log metrics
        self.val_loss.append(loss)
        mean_loss = sum(self.val_loss) / len(self.val_loss)
        self.log(f"val/{self.criterion_type}_loss", mean_loss, prog_bar=True)

        self.val_pixel_acc.append(pixel_accuracy(logits, targets))
        px_acc = sum(self.val_pixel_acc) / len(self.val_pixel_acc)
        self.log("val/pixel_accuracy", px_acc, prog_bar=True)

        self.val_mIoU.append(mean_iou(logits, targets, self.nc))
        miou = sum(self.val_mIoU) / len(self.val_mIoU)
        self.log("val/meanIoU", miou, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_loss = []
        self.val_pixel_acc = []
        self.val_mIoU = []

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
