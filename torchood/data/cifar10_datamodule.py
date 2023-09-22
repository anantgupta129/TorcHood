import lightning
import torch
import torchvision.transforms as transforms
from torchvision import datasets


def create_transforms(variational_auto_encoder=True, train_batch_size=512, val_batch_size=1):
    # Train data transformations
    if variational_auto_encoder:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
                ),
            ]
        )

        # Test data transformations
        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
                ),
            ]
        )
        return train_transforms, test_transforms


class CIFAR10DataModule(lightning.pytorch.LightningDataModule):
    def __init__(
        self, dir="../data", variational_auto_encoder=True, train_batch_size=512, val_batch_size=1
    ):
        super().__init__()
        self.train_transform, self.test_transform = create_transforms(
            variational_auto_encoder=variational_auto_encoder
        )
        self.train_data = datasets.CIFAR10(
            dir, train=True, download=True, transform=self.train_transform
        )
        self.test_data = datasets.CIFAR10(
            dir, train=True, download=True, transform=self.test_transform
        )

        # dataloader arguments - something you'll fetch these from cmdprmt
        train_dataloader_args = dict(
            shuffle=True, batch_size=train_batch_size, num_workers=0, pin_memory=True
        )
        test_dataloader_args = dict(
            shuffle=True, batch_size=val_batch_size, num_workers=0, pin_memory=True
        )
        # train dataloader
        self.train_loader = torch.utils.data.DataLoader(self.train_data, **train_dataloader_args)

        # test dataloader
        self.test_loader = torch.utils.data.DataLoader(self.test_data, **test_dataloader_args)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def train_data(self):
        return self.train_data

    def val_data(self):
        return self.test_data
