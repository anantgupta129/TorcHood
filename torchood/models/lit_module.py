from typing import Any, Tuple
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class LitModule(LightningModule):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss() 

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)    
        return optimizer
    
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
        
        self.log(
            "train/loss", self.train_loss, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, prog_bar=True
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

        return loss
