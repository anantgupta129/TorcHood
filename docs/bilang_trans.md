# Training a Transformer from Scratch for Language Translation

In this tutorial, we provide a step-by-step guide to train a transformer model from scratch using the TorcHood repository. The dataset in focus is sourced from [Hugging Face's Opus Books](https://huggingface.co/datasets/opus_books).

## Table of Contents

- [Training a Transformer from Scratch for Language Translation](#training-a-transformer-from-scratch-for-language-translation)
  - [Table of Contents](#table-of-contents)
    - [Step 0: Environment Setup](#step-0-environment-setup)
    - [Step 1: Essential Imports](#step-1-essential-imports)
    - [Step 2: Dataset and Model Initialization](#step-2-dataset-and-model-initialization)
    - [Step 3: Training the Model](#step-3-training-the-model)
  - [Observing Trained Model Results](#observing-trained-model-results)
  - [Final Thoughts](#final-thoughts)

______________________________________________________________________

### Step 0: Environment Setup

Ensure your system meets the following prerequisites:

1. **Repository Cloning**:

   ```bash
   git clone https://github.com/anantgupta129/TorcHood.git
   cd TorcHood
   ```

2. **Package Installation**:

   ```bash
   python setup.py sdist
   pip install .
   ```

______________________________________________________________________

### Step 1: Essential Imports

Load the required libraries and modules:

```python
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import lightning.pytorch as pl

from torchood.models.bilang_module import BiLangLitModule
from torchood.data.opus_datamodule import OpusBooksDataModule
```

______________________________________________________________________

### Step 2: Dataset and Model Initialization

Set up your dataset and configure your model:

📄 Refer to the [sample configuration file for bilang](../torchood/configs/bilang_config.py) for hyperparameters and settings, and adjust as necessary.

```python
from torchood.configs.bilang_config import get_config

config = get_config()
config["batch_size"] = 32  # Update based on your GPU capacity

datamodule = OpusBooksDataModule(config, pin_memory=True)
datamodule.prepare_data()
datamodule.setup()
```

______________________________________________________________________

### Step 3: Training the Model

Start training your model using PyTorch Lightning:

1. **Standard Training Setup**:

```python
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

model = BiLangLitModule(learning_rate=config["lr"],
                        config=config,
                        tokenizer_src=datamodule.tokenizer_src,
                        tokenizer_tgt=datamodule.tokenizer_tgt)

callbacks = [
    RichProgressBar(leave=True),
    LearningRateMonitor(logging_interval="step"),
]

# Optional: Logging with Weights & Biases
logger = WandbLogger(project="Transformers-BiLang")

trainer = pl.Trainer(callbacks=callbacks, max_epochs=10, accelerator="gpu")
trainer.fit(model=model, datamodule=datamodule)
```

______________________________________________________________________

## Observing Trained Model Results

To delve into a detailed performance analysis and visualization of the trained model, visit [Weights & Biases](https://wandb.ai/anantgupta129/Transformers-BiLang/workspace?workspace=user-anantgupta129).

______________________________________________________________________

## Final Thoughts

As you've progressed through this tutorial, you've gained hands-on experience in training a transformer model from scratch using the TorcHood repository. Experiment further by tweaking the configurations and dataset for enhanced results.

Happy Modeling !!
