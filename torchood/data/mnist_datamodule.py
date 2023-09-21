from typing import Optional, Union

import albumentations as A
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .components.mnist import MNIST, make_transform


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        train_batch_size: int = 512,
        val_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        visual_auto_encoders=True,
        train_augments: Union[A.Compose, None] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transforms = make_transform(visual_auto_encoders)
        self.val_transforms = make_transform(visual_auto_encoders)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        # download
        datasets.MNIST(self.hparams.data_dir, train=True, download=True)
        datasets.MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        # load  only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = datasets.MNIST(self.hparams.data_dir, train=True)
            self.data_val = datasets.MNIST(self.hparams.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=MNIST(self.data_train, self.train_transforms),
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=MNIST(self.data_val, self.val_transforms),
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
