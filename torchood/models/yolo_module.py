from typing import Any, List, Optional, Union

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ..utils.plotting import plot_couple_examples
from .components.yolo.loss import YoloLoss
from .components.yolo.utils import (
    cells_to_bboxes,
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
    non_max_suppression,
)
from .components.yolo.yolov3 import YOLOv3


class YOLOv3LitModule(LightningModule):
    def __init__(
        self, learning_rate: float, config: Any, class_labels: List[str], in_channels: int = 3
    ) -> None:
        super().__init__()

        self.threshold = config.CONF_THRESHOLD
        self.NMS_IOU_THRESH = config.NMS_IOU_THRESH
        self.ANCHORS = config.ANCHORS
        self.S = config.S
        self.CONF_THRESHOLD = config.CONF_THRESHOLD
        self.MAP_IOU_THRESH = config.MAP_IOU_THRESH
        self.NUM_CLASSES = config.NUM_CLASSES
        self.WEIGHT_DECAY = config.WEIGHT_DECAY
        self.scaled_anchors = None
        self.class_labels = class_labels

        self.net: torch.nn.Module = YOLOv3(num_classes=self.NUM_CLASSES, in_channels=in_channels)

        self.learning_rate = learning_rate

        self.loss_fn = YoloLoss()
        self.train_loss = []
        self.val_loss = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.WEIGHT_DECAY
        )
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

    def _step(self, batch: Any) -> torch.Tensor:
        x, y = batch
        y0, y1, y2 = y
        if self.scaled_anchors is None:
            self.scaled_anchors = (
                torch.tensor(self.ANCHORS)
                * torch.tensor(self.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            ).to(y0.device)

        logits = self.forward(x)
        loss = (
            self.loss_fn(logits[0], y0, self.scaled_anchors[0])
            + self.loss_fn(logits[1], y1, self.scaled_anchors[1])
            + self.loss_fn(logits[2], y2, self.scaled_anchors[2])
        )
        return loss  # , logits, y

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss = self._step(batch)
        # update and log metrics
        self.train_loss.append(loss)
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.log("train/loss", mean_loss, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss = []

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss = self._step(batch)
        # update and log metrics
        self.val_loss.append(loss)
        mean_loss = sum(self.val_loss) / len(self.val_loss)
        self.log("val/loss", mean_loss, prog_bar=True, sync_dist=True)

        # plot sample images starts after 10 epoch
        current_epoch = self.current_epoch
        if current_epoch > 10 and (current_epoch%10 == 0 and batch_idx in [0, 10]):
            plotted_image = plot_couple_examples(
                self, batch, 0.6, 0.5, self.scaled_anchors, self.class_labels
            )
            for im in plotted_image:
                self.logger.experiment.add_image(
                    "predictions",
                    torch.tensor(im),
                    f"{self.current_epoch}{batch_idx}",
                )

        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_loss = []
