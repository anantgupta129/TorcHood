<div align="center">

# TorcHood

<img src="./images/logo.jpg"  style="width:20%">


[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.9+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)<br>
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)

__torchood: A User-Friendly Wrapper for Torch, Making Neural Network Training Effortless__ <br> 
`Jumpstart your deep learning endeavors and rapidly prototype custom projects with torchood 🚀⚡🔥`<br>

</div>

## Project Structure 

```bash
├───torchood
│   │   dataset.py
│   │   trainer.py
│   │   __init__.py
│   │
│   ├───models
│   │   │   common.py
│   │   │   custom_resnet.py
│   │   │   mini_resnet.py
│   │   │   resnet.py
│   │   │   __init__.py
│   │
│   ├───utils
│   │       gradcam.py
│   │       misc.py
│   │       plotting.py
│   │       __init__.py
```

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/anantgupta129/TorcHood.git
cd TorcHood

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

! python setup.py sdist
! pip install .
```

## ⚡ Features

- __LR finder__
    ```bash
    from torch.nn import CrossEntropyLoss
    from torchood.utils.misc import find_lr

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    kwargs = {"end_lr":10, "num_iter": 200, "step_mode":"exp"}
    find_lr(model, device, optimizer, criterion=CrossEntropyLoss(), dataloader=train_loader, **kwargs)
    ```

- __grad cam__ 

    ```bash
    from torchood.utils.gradcam import plot_cam_on_image

    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
    plot_cam_on_image(model, [model.layer4[1]], imgs_list, {"mean": mean, "std": std})
    ```

- __Training & training History__
    ```bash
    from torchood.trainer import Trainer
    trainer = Trainer(model, device, optimizer, scheduler)
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        trainer.train(train_loader)
        trainer.evaluate(test_loader)
    
    trainer.plot_history()
    ```

- Supports the CIFAR10 dataset as a sample. In the future, we plan to add support for additional    datasets.
- Supports sample models


## 🤝 Contributing

Contributions are invited! Don't hesitate to submit a pull request.
