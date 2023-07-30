from typing import Optional

import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .components.cifar10 import CIFAR10, make_transform


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_augments: A.Compose | None = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transforms = (
            make_transform("train") if train_augments is None else train_augments
        )
        self.val_transforms = make_transform("val")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        # download
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        # load  only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = datasets.CIFAR10(self.hparams.data_dir, train=True)
            self.data_val = datasets.CIFAR10(self.hparams.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=CIFAR10(self.data_train, self.train_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=CIFAR10(self.data_val, self.val_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
