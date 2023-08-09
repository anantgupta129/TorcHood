import io
from random import randint
from typing import Any

import albumentations as A
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .box_utils import cells_to_bboxes, non_max_suppression


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
        img = img.unsqueeze(0)  # adding one dimention
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


def plot_image(image, boxes, class_labels):
    """Plots predicted bounding boxes on the image."""
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    buffer = io.BytesIO()
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Save the figure to the buffer in png format
    fig.savefig(buffer, format="png")
    # Seek back to the beginning of the buffer
    buffer.seek(0)

    # Read the buffer into a NumPy array
    image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    # Reshape the image to match the dimensions of the figure
    width, height, _ = fig.canvas.get_width_height()
    image = image.reshape(height, width, 4)  # 4 channels (png)

    return image


def plot_couple_examples(model, loader, thresh, iou_thresh, anchors, class_labels):
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # model.train()

    plotted_images = []
    for i in range(batch_size // 4):
        nms_boxes = non_max_suppression(
            bboxes[i],
            iou_threshold=iou_thresh,
            threshold=thresh,
            box_format="midpoint",
        )
        image = plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes, class_labels)
        plotted_images.append(image)

    return plotted_images
