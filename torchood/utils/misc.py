from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder


def per_class_accuracy(model: Any, device: torch.device, data_loader: DataLoader):
    model = model.to(device)
    model.eval()

    classes = data_loader.dataset.classes
    nc = len(classes)
    class_correct = list(0.0 for i in range(nc))
    class_total = list(0.0 for i in range(nc))
    with torch.inference_mode():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(nc):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("[x] Accuracy of ::")
    for i in range(nc):
        print("\t[*] %8s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))


def find_lr(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: Any,
    dataloader: DataLoader,
    **kwargs,
):
    """
    Finds the learning rate for a given model using the LR Finder technique.

    Args:
        model (nn.Module): The model to find the learning rate for.
        device (torch.device): The device to run the model on.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        criterion (Any): The loss function used for training the model.
        dataloader (DataLoader): The data loader used for training the model.
    """

    # Create an instance of the LR Finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    # Run the range test to find the optimal learning rate
    lr_finder.range_test(dataloader, **kwargs)

    # Plot the loss-learning rate graph for inspection
    lr_finder.plot()

    # Reset the model and optimizer to their initial state
    lr_finder.reset()
