import os
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (LearningRateFinder,
                                         LearningRateMonitor, RichProgressBar)


def auto_find_lr_and_fit(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    lr_finder_args: dict,
    trainer_args: dict,
    save_traced_model: bool = False,
    input_shape: tuple = (1, 3, 32, 32),
) -> Any:
    callbacks = [
        RichProgressBar(leave=True),
        LearningRateFinder(**lr_finder_args),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(callbacks=callbacks, **trainer_args)
    trainer.fit(model=model, datamodule=datamodule)

    model = trainer.model
    if save_traced_model:
        model.to_torchscript(
            method="trace", example_inputs=torch.rand(input_shape), file_path="model.traced.pt"
        )
        cwd = os.getcwd()
        print(f" [x] Traced model saved at {cwd}/model.traced.pt")

    return model, trainer
