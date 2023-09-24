from typing import Any, List, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def make_transform(image_set: str) -> A.Compose:
    mean = (0.1307,)
    std = (0.3081,)
    if image_set == "train":
        return A.Compose(
            [
                A.Sequential(
                    [
                        A.CropAndPad(
                            px=4, keep_size=False
                        ),  # padding of 2, keep_size=True by default
                        A.RandomCrop(28, 28),
                    ]
                ),
                A.HorizontalFlip(),
                # A.ShiftScaleRotate(),
                A.CoarseDropout(1, 16, 16, 1, 16, 16, fill_value=mean, mask_fill_value=None),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )


class MNIST(Dataset):
    def __init__(self, ds: Any, transform: A.Compose):
        self.ds = ds
        self.transforms = transform

    @property
    def classes(self) -> List[str]:
        return self.ds.classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i) -> Tuple[torch.Tensor, int]:
        image, label = self.ds[i]
        # apply augmentation
        image = self.transforms(image=np.array(image))["image"]
        return image, label
