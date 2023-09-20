from typing import Any, List, Tuple

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class MNISTToRGB:
    def __call__(self, img):
        img = torch.cat([img, img, img], dim=0)  # Duplicate the single channel into three channels
        return img


def make_transform(visual_auto_encoders: bool) -> A.Compose:
    mean = (0.1307, 0.1307, 0.1307)
    std = (0.3081, 0.3081, 0.3081)
    # Train data transformations
    if visual_auto_encoders:
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                MNISTToRGB(),  # Convert to 3 channels
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
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
