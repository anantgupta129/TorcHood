from typing import Any

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class OpusBooksDataModule(LightningDataModule):
    def __init__(
        self, input_file_path: str, config: Any, num_workers: int = 0, pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        with open(input_file_path, encoding="utf-8") as f:
            self.text = f.read()
        # here are all the unique characters that occur in this text
        chars = sorted(list(set(self.text)))
        vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [
            stoi[c] for c in s
        ]  # encoder: take a string, output a list of integers
        self.decode = lambda line: "".join(
            [itos[i] for i in line]
        )  # decoder: take a list of integers, output a string
        # Train and test splits
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return DataLoader(
    #         dataset=self.train_data,
    #         batch_size=self.hparams.config["batch_size"],
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=True,
    #     )

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(
    #         dataset=self.val_data,
    #         batch_size=1,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=True,
    #     )
