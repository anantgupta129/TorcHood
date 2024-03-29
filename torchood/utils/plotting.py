import io
import random
from random import randint
from typing import Any, List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..models.components.yolo.utils import cells_to_bboxes, non_max_suppression


def plot_sampledata(loader: DataLoader):
    batch_data, batch_label = next(iter(loader))

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].permute(1, 2, 0).numpy())
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

    plt.show()


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    cols, rows = 4, 6
    figure = plt.figure(figsize=(20, 20))
    for i in range(1, cols * rows + 1):
        k = np.random.randint(0, len(loader.dataset))  # random points from test dataset

        img, label = loader.dataset[k]  # separate the image and label
        img = img.unsqueeze(0)  # adding one dimension
        pred = model(img.to(device))  # Prediction

        figure.add_subplot(rows, cols, i)  # making the figure
        plt.title(f"Predcited label {pred.argmax().item()}\n True Label: {label}")  # title of plot
        plt.axis("off")  # hiding the axis
        plt.imshow(img.squeeze(), cmap="gray")  # showing the plot

    plt.show()


def plot_misclassified(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    transformations: A.Compose,
    title: str = "Misclassified (pred/ truth)",
    num_misclf: int = 10,
    return_imgs: bool = False,
):
    count = 1
    rows, cols = int(num_misclf / 5), 5
    figure = plt.figure(figsize=(cols * 3, rows * 3))

    classes = data_loader.dataset.classes
    dataset = data_loader.dataset.ds

    model = model.to(device)
    model.eval()
    imgs_list = []
    with torch.inference_mode():
        while True:
            k = randint(0, len(dataset))
            img, label = dataset[k]
            img = np.array(img)

            aug_img = transformations(image=img)["image"]
            pred = model(aug_img.unsqueeze(0).to(device)).argmax().item()  # Prediction
            if pred != label:
                imgs_list.append(img.copy())

                figure.add_subplot(rows, cols, count)  # adding sub plot
                plt.title(f"{classes[pred]} / {classes[label]}")  # title of plot
                plt.axis("off")
                plt.imshow(img)

                count += 1
                if count == num_misclf + 1:
                    break

    plt.suptitle(title, fontsize=15)
    plt.show()

    if return_imgs:
        return imgs_list


def draw_predictions(image: np.ndarray, boxes: List[List], class_labels: List[str]) -> np.ndarray:
    """Plots predicted bounding boxes on the image."""

    colors = [[random.randint(0, 255) for _ in range(3)] for name in class_labels]

    im = np.array(image)
    height, width, _ = im.shape
    bbox_thick = int(0.6 * (height + width) / 600)

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        conf = box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        x1 = int(upper_left_x * width)
        y1 = int(upper_left_y * height)

        x2 = x1 + int(box[2] * width)
        y2 = y1 + int(box[3] * height)

        cv2.rectangle(
            image, (x1, y1), (x2, y2), color=colors[int(class_pred)], thickness=bbox_thick
        )
        text = f"{class_labels[int(class_pred)]}: {conf:.2f}"
        t_size = cv2.getTextSize(text, 0, 0.7, thickness=bbox_thick // 2)[0]
        c3 = (x1 + t_size[0], y1 - t_size[1] - 3)

        cv2.rectangle(image, (x1, y1), c3, colors[int(class_pred)], -1)
        cv2.putText(
            image,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            bbox_thick // 2,
            lineType=cv2.LINE_AA,
        )

    return image


def plot_couple_examples(model, batch, thresh, iou_thresh, anchors, class_labels):
    x, y = batch
    out = model(x)
    bboxes = [[] for _ in range(x.shape[0])]
    for i in range(3):
        batch_size, A, S, _, _ = out[i].shape
        anchor = anchors[i]
        boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    plotted_images = []
    for i in range(batch_size // 4):
        nms_boxes = non_max_suppression(
            bboxes[i],
            iou_threshold=iou_thresh,
            threshold=thresh,
            box_format="midpoint",
        )
        image = draw_predictions(
            x[i].permute(1, 2, 0).detach().cpu().numpy().copy(), nms_boxes, class_labels
        )
        plotted_images.append(image)

    return plotted_images


def plot_vae_examples(train_loader, classes, vae, mean, std_data, num_of_imges=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classmaps = {i: j for i, j in enumerate(classes)}
    output_images = []
    figure(figsize=(8, 3), dpi=300)
    # Initialize a DataLoader iterator
    data_iter = iter(train_loader)

    # Loop through the DataLoader to generate and collect the images
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Use the 'next' function to get the labels of the next batch
        original_label = labels
        try:
            next_batch = next(data_iter)
            labels = next_batch[1]
        except StopIteration:
            # Handle the case where there are no more batches
            labels = None
        if original_label != labels:
            # Z COMES FROM NORMAL(0, 1)

            # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            z = images
            y = labels
            y = F.one_hot(y, num_classes=10)
            # SAMPLE IMAGES
            x_encoded = vae.encoder(z.to(device), y.to(device))
            # x_encoded = torch.cat((x_encoded, one_hot_labels.to('cuda')), dim=1) #get OHE for label features
            mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            # print(z.shape,y.shape)
            with torch.no_grad():
                pred = vae.decoder(z.to(device), y.to(device)).to(device)

            # UNDO DATA NORMALIZATION
            # normalize = cifar10_normalization()
            mean, std_data = np.array(mean), np.array(std_data)
            img = make_grid(pred).permute(1, 2, 0).cpu().numpy() * std_data + mean
            output_images.append((img, original_label, labels))

            # Check if you have collected 25 images, and if so, break the loop
            if len(output_images) >= num_of_imges:
                break

    # Create a subplot grid for displaying the collected images
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # Plot each image in the grid and add labels horizontally before the image
    for i in range(num_of_imges):
        ax = axes[i // 5, i % 5]
        img, original_label, next_labels = output_images[i]
        # Add original label and next label as text before the image
        label_text = f"Original Label: {classmaps[original_label.item()]} \n"
        if next_labels is not None:
            label_text += f"Input Label: {classmaps[next_labels[0].item()]}"

        ax.text(0, -1.0, label_text, fontsize=8)
        ax.imshow(img)
        ax.axis("off")
    # Show the plot
    plt.show()
