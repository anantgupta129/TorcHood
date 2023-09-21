import lightning
import torch
import torchvision.transforms as transforms
from torchvision import datasets


class MNISTToRGB:
    def __call__(self, img):
        img = torch.cat([img, img, img], dim=0)  # Duplicate the single channel into three channels
        return img


def create_transforms(variational_auto_encoder=True, train_batch_size=512, val_batch_size=1):
    # Train data transformations
    if variational_auto_encoder:
        train_transforms = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                MNISTToRGB(),  # Convert to 3 channels
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
            ]
        )

        # Test data transformations
        test_transforms = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                MNISTToRGB(),  # Convert to 3 channels
                transforms.Normalize((0.1407, 0.1407, 0.1407), (0.4081, 0.4081, 0.4081)),
            ]
        )
        return train_transforms, test_transforms

    else:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081)),
            ]
        )

        # Test data transformations
        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1407), (0.4081)),
            ]
        )
        return train_transforms, test_transforms


class MNISTDataModule(lightning.pytorch.LightningDataModule):
    def __init__(
        self, dir="../data", variational_auto_encoder=True, train_batch_size=512, val_batch_size=1
    ):
        super().__init__()
        self.train_transform, self.test_transform = create_transforms(
            variational_auto_encoder=variational_auto_encoder
        )
        train_data = datasets.MNIST(dir, train=True, download=True, transform=self.train_transform)
        test_data = datasets.MNIST(dir, train=True, download=True, transform=self.test_transform)

        # dataloader arguments - something you'll fetch these from cmdprmt
        train_dataloader_args = dict(
            shuffle=True, batch_size=train_batch_size, num_workers=0, pin_memory=True
        )
        test_dataloader_args = dict(
            shuffle=True, batch_size=val_batch_size, num_workers=0, pin_memory=True
        )
        # train dataloader
        self.train_loader = torch.utils.data.DataLoader(train_data, **train_dataloader_args)

        # test dataloader
        self.test_loader = torch.utils.data.DataLoader(test_data, **test_dataloader_args)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader
