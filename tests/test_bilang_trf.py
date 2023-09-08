# import warnings
# warnings.filterwarnings("ignore")

import os

import lightning.pytorch as pl
import torch

from torchood.configs.bilang_config import get_config
from torchood.data.opus_datamodule import OpusBooksDataModule


def test_dataloader():
    config = get_config()
    config["batch_size"] = 8

    datamodule = OpusBooksDataModule(config)
    datamodule.prepare_data()
    datamodule.setup("")

    for i, batch in enumerate(datamodule.train_dataloader()):
        # print(batch)
        print("==================")
        print(type(batch))
        print(batch["encoder_input"].shape)
        print(batch["decoder_input"].shape)
        print(batch["encoder_mask"].shape)
        print(batch["decoder_mask"].shape)

        if i == 3:
            break

        assert batch["encoder_input"].shape == 8, "batch size should be 8"
        assert batch["decoder_input"].shape == 8, "batch size should be 8"
