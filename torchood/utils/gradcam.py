from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (deprocess_image, preprocess_image,
                                          show_cam_on_image)


def plot_cam_on_image(
    model: Any, target_layer: Any, imgs_list: list, preprocess_args: dict, **kwargs
):
    rows, cols = int(len(imgs_list) / 5), 5
    figure = plt.figure(figsize=(cols * 2, rows * 2))

    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available())
    cam.batch_size = 32

    for i, img in enumerate(imgs_list):
        rgb_img = np.float32(img) / 255
        input_tensor = preprocess_image(rgb_img, **preprocess_args)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None, **kwargs)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        figure.add_subplot(rows, cols, i + 1)  # adding sub plot
        plt.axis("off")  # hiding the axis
        plt.imshow(cam_image, cmap="rainbow")  # showing the plot

    plt.tight_layout()
    plt.show()
