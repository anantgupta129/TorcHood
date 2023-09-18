"""Download dataset : https://www.robots.ox.ac.uk/~vgg/data/pets/
"""

import os
from typing import List, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def make_transform(image_set: str) -> A.Compose:
    """Creates a transformation pipeline based on the given image set ("train" or "test").

    Args:
    - image_set (str): The set of images for which transformations need to be made ("train" or otherwise).

    Returns:
    - A.Compose: A composition of various augmentations.
    """
    # Mean and Standard deviation for normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Augmentations for training set
    if image_set == "train":
        return A.Compose(
            [
                A.Rotate(limit=35),  # Rotation augmentation
                A.HorizontalFlip(),  # Horizontal flip augmentation
                A.VerticalFlip(),  # Vertical flip augmentation
                A.Resize(height=256, width=256),  # Resizing the images
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),  # Normalizing images
                ToTensorV2(),  # Converting images to PyTorch tensors
            ],
        )
    # Augmentations for validation or test set
    else:
        return A.Compose(
            [
                A.Resize(height=256, width=256),  # Resizing the images
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),  # Normalizing images
                ToTensorV2(),  # Converting images to PyTorch tensors
            ],
        )


class OxfordIiitPets(Dataset):
    """PyTorch Dataset for Oxford-IIIT Pet Dataset.

    Args:
    - image_dir (str): Path to the directory containing images.
    - images_list (List[str]): List of paths to the images.
    - mask_dir (str): Path to the directory containing masks.
    - transforms (Union[A.Compose, None]): Transformations to apply on the images and masks.
    """

    def __init__(
        self,
        image_dir: str,
        images_list: List[str],
        mask_dir: str,
        transforms: Union[A.Compose, None] = None,
    ):
        self.image_dir = image_dir
        self.images_list = images_list
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches an image and its corresponding mask based on the index provided.

        Args:
        - idx (int): Index of the image and mask pair to fetch.

        Returns:
        - tuple: A tuple containing the transformed image and mask.
        """
        ip = os.path.join(self.image_dir, self.images_list[idx])
        mp = os.path.join(self.mask_dir, self.images_list[idx].replace("jpg", "png"))

        image = Image.open(ip).convert("RGB")
        mask = Image.open(mp)
        if self.transforms:
            transformed = self.transforms(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


# if __name__=="__main__":
#     import glob

#     import matplotlib.pyplot as plt
#     from torch.utils.data import DataLoader

#     image_dir = "sample_dataset/oxford_iiit_pets/images"
#     mask_dir = "sample_dataset/oxford_iiit_pets/annotations/trimaps"

#     images_list = []
#     for ext in [".jpg", ".jpeg", ".png"]: # filtering images
#         files = glob.glob(f"{image_dir}/*{ext}")
#         images_list += [os.path.basename(f) for f in files]

#     dset = OxfordIiitPets(
#         mask_dir=mask_dir,
#         image_dir=image_dir,
#         images_list=images_list,
#         transforms=make_transform(image_set="val"),
#     )

#     loader = DataLoader(dset, batch_size=1)
#     for x, y in loader:
#         print(x.shape, y.shape)
#         plt.subplot(1, 2, 1)
#         plt.imshow(x[0].permute(1, 2, 0))

#         plt.subplot(1, 2, 2)
#         plt.imshow(y[0])

#         plt.show()
