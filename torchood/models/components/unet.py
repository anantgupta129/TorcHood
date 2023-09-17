"""Implementation of the U-Net architecture.

U-Net is designed for fast and precise segmentation of images.
Paper: https://arxiv.org/abs/1505.04597

Attributes:
    ContractingBlock: A class that represents the contracting (downsampling) layers of the U-Net.
    ExpandingBlock: A class that represents the expanding (upsampling) layers of the U-Net.
    UNet: A class that represents the U-Net architecture.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the block layers."""
        x = self.layer1(x)
        if self.strided_conv:
            skip = x
            x = self.layer2(x)
        else:
            x = self.layer2(x)
            skip = x
            x = self.maxpool(x)

        return x, skip


class ExpandingBlock(nn.Module):
    """Defines a block for the expanding path. Decoder block.

    Attributes:
        conv1, conv2 (torch.nn.Conv2d): Convolutional layers.
        bn1, bn2 (torch.nn.BatchNorm2d): Batch normalization layers.
        relu1, relu2 (torch.nn.ReLU): ReLU activation layers.
        upsample (torch.nn.ConvTranspose2d): Upsampling layer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Upsampling layer to double the spatial dimensions
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the block layers and concatenates with the skip
        connection."""
        x = self.upsample(x)
        # Resize x to match the spatial dimensions of skip tensor
        if x.shape[2] != skip.shape[2]:
            x = F.resize(x, size=skip.shape[2:])
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

    def __init__(self, in_channels: int, out_channels: int, strided_conv: bool = False):
        super().__init__()

        # Contracting Path (Encoder)
        self.contract1 = ContractingBlock(in_channels, 64, strided_conv)
        self.contract2 = ContractingBlock(64, 128, strided_conv)
        self.contract3 = ContractingBlock(128, 256, strided_conv)
        self.contract4 = ContractingBlock(256, 512, strided_conv)
        self.contract5 = ContractingBlock(512, 1024)

        # Expanding Path (Decoder)
        self.expand1 = ExpandingBlock(1024, 512)
        self.expand2 = ExpandingBlock(512, 256)
        self.expand3 = ExpandingBlock(256, 128)
        self.expand4 = ExpandingBlock(128, 64)

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


# if __name__ == "__main__":
#     from torchinfo import summary

#     # Creating a U-Net model
#     model = UNet(3, 10, strided_conv=True)
#     input_size = (1, 3, 572, 572)
#     x = torch.randn(*input_size)

#     out = model(x)  # dry run
#     print(out.shape)
#     # Printing the model summary
#     summary(model, input_size=input_size, device="cpu")
