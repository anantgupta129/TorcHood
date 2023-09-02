# Train Transformer form scratch for language translation

In this tutorial, we will guide you step-by-step on how to train transformer from scratch using the TorcHood repository. We will be utilizing the a language dataset which contains itlaian and english text, which can be found [here](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video).

## Prerequisites

- Basic understanding of Python.
- Familiarity with Deep Learning concepts and PyTorch Lightning.

## Table of Contents

- [Train Transformer form scratch for language translation](#train-yolov3-from-scratch-using-lightning-on-pascalvoc-dataset)
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

from torchood.configs import bilang_config
from torchood.models.bilang_module import  BiLangLitModule
from torchood.data.opus_datamodule import OpusBooksDataModulee
```

______________________________________________________________________

### Step 2: Loading Dataset and Model

Here, we'll set up the data and model configuration.

📄 For your convenience, check out the [sample config file for yolov3](../torchood/configs/yolo_config.py). This file contains hyperparameters and settings. You can modify it to your liking

```python
dataloader_config = bilang_config.get_config()
# dataloader_config["batch_size"] = 8
# dataloader_config["num_workers"] = 8

datamodule = OpusBooksDataModule(dataloader_config,pin_memory=True)
datamodule.prepare_data()
datamodule.setup(" ")
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

model = BiLangLitModule(learning_rate=1e-3,
                        config=dataloader_config,
                        tokenizer_src=datamodule.tokenizer_src,
                        tokenizer_tgt=datamodule.tokenizer_tgt)

callbacks = [
    RichProgressBar(leave=True),
    LearningRateMonitor(logging_interval="step"),
]
logger = WandbLogger(project="transformer from scratch") # logging with weights & biases (optional)
trainer = pl.Trainer(callbacks=callbacks, max_epochs=10,
                    accelerator="gpu")
trainer.fit(model=model, datamodule=datamodule)

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
