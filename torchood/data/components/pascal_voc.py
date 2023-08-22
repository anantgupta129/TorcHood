"""Creates a Pytorch dataset to load the Pascal VOC."""

import os
import random
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from ...models.components.yolo.utils import iou_width_height as iou
from ...models.components.yolo.utils import xywhn2xyxy, xyxy2xywhn

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        label_dir: str,
        anchors: List[List[float]],
        image_size: int = 416,
        transform=None,
        mosaic_prob: float = 0.8,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.mosaic_border = [image_size // 2, image_size // 2]
        self.transform = transform
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.mosaic_prob = mosaic_prob

    def __len__(self):
        return len(self.annotations)

    def load_mosaic(self, index: int) -> np.ndarray:
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        s = self.image_size
        yc, xc = (
            int(random.uniform(x, 2 * s - x)) for x in self.mosaic_border
        )  # mosaic center x, y
        indices = [index] + random.choices(range(len(self)), k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            bboxes = np.roll(
                np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
            ).tolist()
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            img = np.array(Image.open(img_path).convert("RGB"))

            h, w = img.shape[0], img.shape[1]
            labels = np.array(bboxes)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full(
                    (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if labels.size:
                labels[:, :-1] = xywhn2xyxy(
                    labels[:, :-1], w, h, padw, padh
                )  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, :-1],):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate
        labels4[:, :-1] = xyxy2xywhn(labels4[:, :-1], 2 * s, 2 * s)
        labels4[:, :-1] = np.clip(labels4[:, :-1], 0, 1)
        labels4 = labels4[labels4[:, 2] > 0]
        labels4 = labels4[labels4[:, 3] > 0]
        return img4, labels4

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, tuple]:
        if random.random() < self.mosaic_prob:
            image, bboxes = self.load_mosaic(index)
        else:
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            bboxes = np.roll(
                np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
            ).tolist()
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        return image, bboxes

    def make_targets(self, bboxes: List[List], scales: List[int]) -> List[torch.Tensor]:
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        target = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in scales]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = scales[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = target[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    target[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    target[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    target[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    target[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return target

    def collate_fn(self, batch: Any) -> Tuple[torch.Tensor, Any]:
        image, bboxes = zip(*batch)
        image = torch.stack(image, 0)

        imgsz = image.shape[-1]
        scales = [imgsz // 32, imgsz // 16, imgsz // 8]
        targets_1 = []
        targets_2 = []
        targets_3 = []
        for box in bboxes:
            target1, target2, target3 = self.make_targets(box, scales)
            targets_1.append(target1)
            targets_2.append(target2)
            targets_3.append(target3)

        targets_1 = torch.stack(targets_1, 0)
        targets_2 = torch.stack(targets_2, 0)
        targets_3 = torch.stack(targets_3, 0)

        return image, [targets_1, targets_2, targets_3]

    def collate_fn4(self, batch: Any) -> Tuple[torch.Tensor, Any]:
        image, bboxes = zip(*batch)

        if random.random() > 0.2:
            size = random.choice([352, 384, 480, 832])
            resized_imzs = []
            for img in image:
                resized_im = F.interpolate(
                    img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
                )
                resized_imzs.append(resized_im)
            image = torch.stack(resized_imzs, 0)
        else:
            image = torch.stack(image, 0)

        imgsz = image.shape[-1]
        scales = [imgsz // 32, imgsz // 16, imgsz // 8]
        targets_1 = []
        targets_2 = []
        targets_3 = []
        for box in bboxes:
            target1, target2, target3 = self.make_targets(box, scales)
            targets_1.append(target1)
            targets_2.append(target2)
            targets_3.append(target3)

        targets_1 = torch.stack(targets_1, 0)
        targets_2 = torch.stack(targets_2, 0)
        targets_3 = torch.stack(targets_3, 0)

        return image, [targets_1, targets_2, targets_3]


# def test():
#     import matplotlib.pyplot as plt

#     import torchood.configs.yolo_config as config
#     from torchood.models.components.yolo.utils import (
#         cells_to_bboxes,
#         non_max_suppression,
#     )
#     from torchood.utils.plotting import draw_predictions

#     anchors = config.ANCHORS

#     transform = config.test_transforms

#     dataset = YOLODataset(
#         r"C:\Users\anant\Downloads\archive\PASCAL_VOC/train.csv",
#         r"C:\Users\anant\Downloads\archive\PASCAL_VOC/images/",
#         r"C:\Users\anant\Downloads\archive\PASCAL_VOC/labels/",
#         S=[13, 26, 52],
#         anchors=anchors,
#         transform=transform,
#     )

#     loader = DataLoader(
#         dataset=dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn4
#     )
#     for x, y in loader:
#         boxes = []
#         print(x.shape)
#         print(len(y), type(y))
#         print(y[0].shape, y[1].shape, y[2].shape)
#         # break
#         imgsz = x.shape[-1]
#         S = [imgsz // 32, imgsz // 16, imgsz // 8]
#         scaled_anchors = torch.tensor(anchors) / (
#             1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#         )
#         for i in range(y[0].shape[1]):
#             anchor = scaled_anchors[i]
#             # print(anchor.shape)
#             # print(y[i].shape)
#             boxes += cells_to_bboxes(y[i], is_preds=False, S=y[i].shape[2], anchors=anchor)
#         for im, box in zip(x, boxes):
#             boxes = non_max_suppression(box, iou_threshold=1, threshold=0.7, box_format="midpoint")
#             print(im.shape)
#             np_img = im.squeeze().permute(1, 2, 0).numpy().copy()
#             np_img = draw_predictions(np_img, boxes, config.PASCAL_CLASSES)
#             plt.imshow(np_img)
#             plt.show()


# if __name__ == "__main__":
#     test()
