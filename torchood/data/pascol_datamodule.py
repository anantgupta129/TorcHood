from typing import Optional, Any

import albumentations as A
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .components.pascol_voc import YOLODataset


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        config: Any,
        data_dir: str = "./data",
        batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_csv_path=config.DATASET + "/train.csv" 
        self.test_csv_path=config.DATASET + "/test.csv"

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 20
    
    @property
    def class_names(self):
        return self.hparams.config.PASCAL_CLASSES

    def setup(self, stage: str=None) -> None:
        # load  only if not loaded already
        if not self.data_train and not self.data_val:
            IMAGE_SIZE = self.hparams.config.IMAGE_SIZE
            self.data_train = YOLODataset(
                self.hparams.config.train_csv_path,
                transform=self.hparams.config.train_transforms,
                S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
                img_dir=self.hparams.config.IMG_DIR,
                label_dir=self.hparams.config.LABEL_DIR,
                anchors=self.hparams.config.ANCHORS,
            )
            self.data_val = YOLODataset(
                self.hparams.self.hparams.config.test_csv_path,
                transform=self.hparams.config.test_transforms,
                S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
                img_dir=self.hparams.config.IMG_DIR,
                label_dir=self.hparams.config.LABEL_DIR,
                anchors=self.hparams.config.ANCHORS,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.config.BATCH_SIZE,
            num_workers=self.hparams.config.NUM_WORKERS,
            pin_memory=self.hparams.config.PIN_MEMORY,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.config.BATCH_SIZE,
            num_workers=self.hparams.config.NUM_WORKERS,
            pin_memory=self.hparams.config.PIN_MEMORY,
            shuffle=False,
        )
