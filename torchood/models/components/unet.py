"""Implementation of the U-Net architecture.

U-Net is designed for fast and precise segmentation of images.
Paper: https://arxiv.org/abs/1505.04597

Attributes:
    ContractingBlock: A class that represents the contracting (downsampling) layers of the U-Net.
    ExpandingBlock: A class that represents the expanding (upsampling) layers of the U-Net.
    UNet: A class that represents the U-Net architecture.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ContractingBlock(nn.Module):
    """Defines a block for the contracting path. Encoder Block.

    Attributes:
        layer1 (torch.nn.Sequential): First part of the contracting block.
        layer2 (torch.nn.Sequential): Second part of the contracting block.
    """

    def __init__(self, in_channels: int, out_channels: int, strided_conv: bool = False):
        super().__init__()

        # Two convolution layers followed by batch normalization and ReLU activation
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if strided_conv:
            self.layer2 = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.strided_conv = strided_conv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pass through layer 1
        x = self.layer1(x)

        if self.strided_conv:
            # If strided conv is used, save the input tensor as the skip connection
            skip = x
            # Pass through layer 2
            x = self.layer2(x)
        else:
            # Pass through layer 2
            x = self.layer2(x)
            # Save the output tensor as the skip connection
            skip = x
            # Perform max pooling
            x = self.maxpool(x)

        return x, skip


class ExpandingBlock(nn.Module):
    """Defines a block for the expanding path. Decoder block.

    Attributes:
        conv1, conv2 (torch.nn.Conv2d): Convolutional layers.
        bn1, bn2 (torch.nn.BatchNorm2d): Batch normalization layers.
        relu1, relu2 (torch.nn.ReLU): ReLU activation layers.
        upsample (torch.nn.ConvTranspose2d) | (torch.nn.Upsample): Upsampling layer.
    """

    def __init__(self, in_channels: int, out_channels: int, upsample_mode: str = "transpose"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Upsampling layer to double the spatial dimensions
        if upsample_mode == "transpose":
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=False
            )
        elif upsample_mode == "upsample":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            raise ValueError("Upsample mode must be 'transpose' or 'upsample'.")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the block layers and concatenates with the skip
        connection."""
        x = self.upsample(x)
        # Resize x to match the spatial dimensions of skip tensor
        if x.shape[2] != skip.shape[2]:
            print(x.shape, skip.shape)
            x = TF.resize(x, size=skip.shape[2:])
        x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UNet(nn.Module):
    """Implementation of the U-Net architecture.

    The U-Net model consists of a contracting path, a bottom layer, and an expansive path. The
    contracting path captures context, and the expansive path enables precise localization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strided_conv: bool = False,
        upsample_mode: str = "transpose",
    ):
        """Initializes the U-Net architecture.

        Args:
        - in_channels (int): Number of input channels to the U-Net. Typically 3 for RGB images.
        - out_channels (int): Number of output channels from the U-Net. This is usually the number
                              of classes for segmentation tasks.
        - strided_conv (bool, optional): If True, use strided convolution for downsampling instead
                                         of max pooling. Defaults to False.
        - upsample_mode (str, optional): Mode of upsampling: "transpose" for transposed convolution
                                         and "upsample" for bilinear upsampling. Defaults to "transpose".

        Raises:
        - ValueError: If upsample_mode is neither 'transpose' nor 'upsample'.
        """
        super().__init__()

        # Contracting Path (Encoder)
        self.contract1 = ContractingBlock(in_channels, 64, strided_conv)
        self.contract2 = ContractingBlock(64, 128, strided_conv)
        self.contract3 = ContractingBlock(128, 256, strided_conv)
        self.contract4 = ContractingBlock(256, 512, strided_conv)
        self.contract5 = ContractingBlock(512, 1024)

        # Expanding Path (Decoder)
        self.expand1 = ExpandingBlock(1024, 512, upsample_mode)
        self.expand2 = ExpandingBlock(512, 256, upsample_mode)
        self.expand3 = ExpandingBlock(256, 128, upsample_mode)
        self.expand4 = ExpandingBlock(128, 64, upsample_mode)

        # Final convolution to map to the desired number of classes
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the U-Net model and returns the segmentation map."""
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 = self.contract4(x)
        _, x = self.contract5(x)  # in last layer pooling / strides is not required

        # Expanding path with skip connections
        x = self.expand1(x, skip4)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)

        # Final convolution layer
        x = self.final_conv(x)
        return x


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        """Initializes the Dice Loss module.

        Args:
        - eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.
        """
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the Dice Loss between predicted logits and true labels.

        Args:
        - logits (torch.Tensor): Predicted logits from the model. Shape (batch_size, classes, height, width).
        - target (torch.Tensor): Ground truth labels. Shape (batch_size, height, width).

        Returns:
        - torch.Tensor: Computed Dice Loss.
        """

        num_classes = logits.shape[1]
        # Apply sigmoid to logits
        logits = torch.sigmoid(logits)

        # Calculate Dice Loss for each class and then average
        dice_loss = 0.0
        for i in range(num_classes):
            dice_loss += self._dice_loss_per_channel(logits[:, i], (target == i).float())

        return dice_loss / num_classes

    def _dice_loss_per_channel(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (logits_flat * target_flat).sum()
        union = logits_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1 - dice


def pixel_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute the pixel accuracy.

    Args:
    - output (torch.Tensor): The network's output of shape (batch_size, num_classes, height, width).
    - target (torch.Tensor): The ground truth labels of shape (batch_size, height, width).

    Returns:
    - float: Pixel accuracy.
    """
    _, predicted = torch.max(output, 1)
    correct_pixels = (predicted == target).sum().item()
    total_pixels = torch.numel(target)
    accuracy = correct_pixels / total_pixels
    return accuracy


def iou_per_class(output: torch.Tensor, target: torch.Tensor, num_classes: int) -> List:
    """Compute the IoU for each class.

    Args:
    - output (torch.Tensor): The network's output of shape (batch_size, num_classes, height, width).
    - target (torch.Tensor): The ground truth labels of shape (batch_size, height, width).
    - num_classes (int): Number of classes.

    Returns:
    - list of float: IoU for each class.
    """
    _, predicted = torch.max(output, 1)

    iou_list = []
    for cls in range(num_classes):
        intersection = (predicted == cls & target == cls).sum().float().item()
        union = ((predicted == cls) | (target == cls)).sum().float().item()
        if union == 0:
            iou_list.append(np.nan)
        else:
            iou_list.append(intersection / union)
    return iou_list


def mean_iou(output: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Compute the mean IoU.

    Args:
    - output (torch.Tensor): The network's output of shape (batch_size, num_classes, height, width).
    - target (torch.Tensor): The ground truth labels of shape (batch_size, height, width).
    - num_classes (int): Number of classes.

    Returns:
    - float: Mean IoU across all classes.
    """
    iou_list = iou_per_class(output, target, num_classes)
    valid_iou = [iou for iou in iou_list if not np.isnan(iou)]
    mIoU = sum(valid_iou) / len(valid_iou)
    return mIoU


# if __name__ == "__main__":
#     from torchinfo import summary

#     # Creating a U-Net model
#     # model = UNet(3, 10, strided_conv=True)
#     model = UNet(3, 10, strided_conv=True, upsample_mode="upsample")
#     input_size = (1, 3, 256, 256)
#     x = torch.randn(*input_size)

#     out = model(x)  # dry run
#     print(out.shape)
#     # Printing the model summary
#     summary(model, input_size=input_size, device="cpu")
