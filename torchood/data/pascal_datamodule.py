from typing import Any, List, Optional, Union

import albumentations as A
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .components.pascal_voc import YOLODataset


class PascalVOCDataModule(LightningDataModule):
    def __init__(
        self,
        DATASET: str,
        ANCHORS: List[List],
        class_names: List[str],
        IMAGE_SIZE: int = 416,
        batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transforms: Union[A.Compose, None] = None,
        test_transforms: Union[A.Compose, None] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_csv_path = DATASET + "/train.csv"
        self.test_csv_path = DATASET + "/test.csv"
        self.IMG_DIR = DATASET + "/images/"
        self.LABEL_DIR = DATASET + "/labels/"

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    @property
    def num_classes(self):
        return 20

    @property
    def class_names(self):
        return self.hparams.class_names

    def setup(self, stage: str = None) -> None:
        # load  only if not loaded already
        if not self.data_train and not self.data_val:
            IMAGE_SIZE = self.hparams.IMAGE_SIZE
            self.data_train = YOLODataset(
                self.train_csv_path,
                transform=self.train_transforms,
                S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
                img_dir=self.IMG_DIR,
                label_dir=self.LABEL_DIR,
                anchors=self.hparams.ANCHORS,
            )
            self.data_val = YOLODataset(
                self.test_csv_path,
                transform=self.test_transforms,
                S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
                img_dir=self.IMG_DIR,
                label_dir=self.LABEL_DIR,
                anchors=self.hparams.ANCHORS,
                mosaic_prob=0.0,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
