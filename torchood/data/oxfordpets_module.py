import glob
import os
import random
from typing import Optional, Union

import albumentations as A
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from .components.oxfordpets import OxfordIiitPets, make_transform


class OxfordIiitPetsDataModule(LightningDataModule):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_augments: Union[A.Compose, None] = None,
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

    def setup(self, stage=None) -> None:
        # load
        if not self.data_train and not self.data_train:
            images_list = []
            for ext in [".jpg", ".jpeg", ".png"]:  # filtering images
                files = glob.glob(f"{self.hparams.image_dir}/*{ext}")
                images_list += [os.path.basename(f) for f in files]

            random.shuffle(images_list)
            train_images = images_list[: int(len(images_list) * 0.8)]
            val_images = images_list[int(len(images_list) * 0.8) :]

            self.data_train = OxfordIiitPets(
                self.hparams.image_dir,
                train_images,
                self.hparams.mask_dir,
                self.train_transforms,
            )
            self.data_val = OxfordIiitPets(
                self.hparams.image_dir,
                val_images,
                self.hparams.mask_dir,
                self.val_transforms,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
