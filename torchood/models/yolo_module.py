from typing import Any, Optional, Union

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ..utils.box_utils import get_evaluation_bboxes, mean_average_precision
from ..utils.yolo_loss import YoloLoss
from .components.yolov3 import YOLOv3


class Metric:
    def __init__(self) -> None:
        self._load()

    def _load(self) -> None:
        self.losses = []
        self.tot_class_preds, self.correct_class = 0, 0
        self.tot_noobj, self.correct_noobj = 0, 0
        self.tot_obj, self.correct_obj = 0, 0

    def reset(self) -> None:
        self._load()


class YOLOv3LitModule(LightningModule):
    def __init__(self, learning_rate: float, config: Any, in_channels: int = 3) -> None:
        super().__init__()

        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)
        self.threshold = config.CONF_THRESHOLD
        self.iou_threshold = config.NMS_IOU_THRESH
        self.anchors = config.ANCHORS
        self.threshold = config.CONF_THRESHOLD
        self.iou_threshold = config.MAP_IOU_THRESH
        self.num_classes = config.NUM_CLASSES
        self.WEIGHT_DECAY = config.WEIGHT_DECAY

        self.net: torch.nn.Module = YOLOv3(num_classes=self.num_classes, in_channels=in_channels)

        self.learning_rate = learning_rate

        self.loss_fn = YoloLoss()
        self.train_metric = Metric()
        self.val_metric = Metric()

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

    def _step(self, batch: Any, metric: Union[Metric, Any]) -> torch.Tensor:
        x, y = batch
        y0, y1, y2 = y

        logits = self.forward(x)
        loss = (
            self.loss_fn(logits[0], y0, self.scaled_anchors[0])
            + self.loss_fn(logits[1], y1, self.scaled_anchors[1])
            + self.loss_fn(logits[2], y2, self.scaled_anchors[2])
        )
        metric.losses.append(loss)

        for i in range(3):
            y[i] = y[i]
            obj = y[i][..., 0] == 1  # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            metric.correct_class += torch.sum(
                torch.argmax(logits[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            metric.tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(logits[i][..., 0]) > self.threshold
            metric.correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            metric.tot_obj += torch.sum(obj)
            metric.correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            metric.tot_noobj += torch.sum(noobj)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss = self._step(batch, self.train_metric)
        # update and log metrics
        mean_loss = sum(self.train_metric.losses) / len(self.train_metric.losses)
        self.log("train/loss", mean_loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train/class_acc",
            (self.train_metric.correct_class / (self.train_metric.tot_class_preds + 1e-16)) * 100,
        )
        self.log(
            "train/no_obj_acc",
            (self.train_metric.correct_noobj / (self.train_metric.tot_noobj + 1e-16)) * 100,
        )
        self.log(
            "train/obj_acc",
            (self.train_metric.correct_obj / (self.train_metric.tot_obj + 1e-16)) * 100,
        )
        self.train_metric.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss = self._step(batch, self.val_metric)
        # update and log metrics
        mean_loss = sum(self.val_metric.losses) / len(self.val_metric.losses)
        self.log("val/loss", mean_loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val/class_acc",
            (self.val_metric.correct_class / (self.val_metric.tot_class_preds + 1e-16)) * 100,
            prog_bar=True,
        )
        self.log(
            "val/no_obj_acc",
            (self.val_metric.correct_noobj / (self.val_metric.tot_noobj + 1e-16)) * 100,
            prog_bar=True,
        )
        self.log(
            "val/obj_acc",
            (self.val_metric.correct_obj / (self.val_metric.tot_obj + 1e-16)) * 100,
            prog_bar=True,
        )

        pred_boxes, true_boxes = get_evaluation_bboxes(
            self.trainer.val_dataloaders,
            self,
            iou_threshold=self.NMS_IOU_THRESH,
            anchors=self.ANCHORS,
            threshold=self.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=self.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=self.NUM_CLASSES,
        )

        self.log("val/MAP", mapval, prog_bar=True)
        self.val_metric.reset()
