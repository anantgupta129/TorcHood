from typing import Any, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class LitModule(LightningModule):
    def __init__(self, net: torch.nn.Module, num_classes: int, learning_rate: float) -> None:
        super().__init__()

        self.net = net
        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

    def _step(self, batch: Any) -> Tuple[torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, preds, targets = self._step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)

        self.log("train/loss", self.train_loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

        return loss
