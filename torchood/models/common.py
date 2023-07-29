import torch
import torch.nn as nn
from torchinfo import summary


class BaseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def summarize(self, device: torch.device, input_size: tuple = (1, 3, 32, 32)):
        print(summary(self.to(device), input_size=input_size))
