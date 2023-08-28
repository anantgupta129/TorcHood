# Train YoloV3 from Scratch Using Lightning On PascalVOC Dataset

In this tutorial, we will guide you step-by-step on how to train YoloV3 from scratch using the TorcHood repository. We will be utilizing the PascalVOC dataset, which can be found [here](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video).

## Prerequisites

- Basic understanding of Python.
- Familiarity with Deep Learning concepts and PyTorch Lightning.

## Table of Contents

- [Train YoloV3 from Scratch Using Lightning On PascalVOC Dataset](#train-yolov3-from-scratch-using-lightning-on-pascalvoc-dataset)
  - [Prerequisites](#prerequisites)
  - [Table of Contents](#table-of-contents)
    - [Step 0: Installation](#step-0-installation)
    - [Step 1: Imports](#step-1-imports)
    - [Step 2: Loading Dataset and Model](#step-2-loading-dataset-and-model)
    - [Step 3: Start Training](#step-3-start-training)
    - [Step 4: Evaluation](#step-4-evaluation)
  - [View Trained Model Results](#view-trained-model-results)
  - [Conclusions](#conclusions)

______________________________________________________________________

### Step 0: Installation

Before you proceed, ensure you have the necessary dependencies set up:

1. Clone the repository:

   ```bash
   git clone https://github.com/anantgupta129/TorcHood.git
   cd TorcHood
   ```

2. Install the package from the source:

   ```bash
   python setup.py sdist
   pip install .
   ```

______________________________________________________________________

### Step 1: Imports

These are the essential libraries and modules you'll need:

```python
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import lightning.pytorch as pl

from torchood.configs import yolo_config
from torchood.models.yolo_module import  YOLOv3LitModule
from torchood.data.pascal_datamodule import PascalVOCDataModule
```

______________________________________________________________________

### Step 2: Loading Dataset and Model

Here, we'll set up the data and model configuration.

📄 For your convenience, check out the [sample config file for yolov3](../torchood/configs/yolo_config.py). This file contains hyperparameters and settings. You can modify it to your liking

```python
dataloader_config = yolo_config.dataloader_config
dataloader_config["batch_size"] = 32
dataloader_config["num_workers"] = 8

datamodule = PascalVOCDataModule(**dataloader_config)
model = YOLOv3LitModule(learning_rate=1e-3, config=yolo_config, class_labels=yolo_config.PASCAL_CLASSES)
```

______________________________________________________________________

### Step 3: Start Training

Begin training your model using PyTorch Lightning.

1. You can use the standard training setup:

```python
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

callbacks = [
    RichProgressBar(leave=True),
    LearningRateMonitor(logging_interval="step"),
]
logger = WandbLogger(project="YoloV3-Scratch") # logging with weights & biases (optional)
trainer = pl.Trainer(callbacks=callbacks, logger=logger, max_epochs=40,
                    accelerator="gpu", precision=16)
trainer.fit(model=model, datamodule=datamodule)
```

2. Alternatively, leverage [helper](../torchood/utils/helper.py) functions for more automation:

```python
from torchood.utils.helper import auto_find_lr_and_fit
# This method will automatically find the best learning rate and start the training
trainer_args = {
    'accelerator': "gpu",
    'precision': '16',
    'max_epochs': 40,
}
trainer, model = auto_find_lr_and_fit(model, datamodule, trainer_args=trainer_args)

# without finding best lr
from torchood.utils.helper import train_lightning
trainer, model = train_lightning(model, datamodule, trainer_args=trainer_args)

```

______________________________________________________________________

### Step 4: Evaluation

Post-training, evaluate the performance of the model:

```python
from torchood.models.components.yolo.utils import (
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
)

model.to("cuda")

# Evaluate accuracy on the training set
check_class_accuracy(model, datamodule.train_dataloader(), yolo_config.CONF_THRESHOLD)

# Evaluate accuracy on the validation set
loader = datamodule.val_dataloader()
check_class_accuracy(model, loader, yolo_config.CONF_THRESHOLD)

# Calculate mean average precision (MAP)
pred_boxes, true_boxes = get_evaluation_bboxes(
    loader,
    model,
    iou_threshold=yolo_config.NMS_IOU_THRESH,
    anchors=yolo_config.ANCHORS,
    threshold=yolo_config.CONF_THRESHOLD,
)

mapval = mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=yolo_config.MAP_IOU_THRESH,
    box_format="midpoint",
    num_classes=yolo_config.NUM_CLASSES,
)
print("MAP: ", mapval.item())
```

______________________________________________________________________

## View Trained Model Results

- For a detailed breakdown of the model's performance and visualizations, check the results on [Weights & Biases](https://api.wandb.ai/links/anantgupta129/83aopx49).

______________________________________________________________________

## Conclusions

By now, you should have successfully trained a YOLOv3 model from scratch using PyTorch Lightning on the PascalVOC dataset. This tutorial aimed to guide you seamlessly through the process.

Happy modeling!
