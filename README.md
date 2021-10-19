# TorVizLib


Contains the utils, models etc for building vision based for training and vizualizing pytorch CNN models

```
├── dataloaders
│   ├── CIFAR10.py
│   ├── tiny_image_net.py
│ 
├── models
│   ├── custom_resnet.py
│   └── resnet.py
│  
├── utils
│    ├── datastats.py
│    ├── dump_list.py
│    ├── gradcam.py
│    ├── lr_finder.py
│    └── plotting.py
│ 
│   
├── LICENSE
├── main.py
└── README.md
```

USAGE
-----------

- Dataloaders contains lodaer function to laod data for pytorch Currently for Tiny Image Net 200 and CIDAR10 dataset
- Models contains custom ResNet, ResNet 18 and ResNET 34
- utils has functon used for  Gradcam and Vizulation , Finding Learning Rate for dataset, and saving history locally
- main.py is used for traininf, testing and saving the model
```
main.py

  def fit_model(net, optimizer, device, NUM_EPOCHS,train_loader, test_loader, use_l1=False, scheduler=None, save_best=False):
    """Fit the model
    Args:
        net (instance): torch model instance of defined model
        optimizer (function): optimizer to be used
        device (str): "cpu" or "cuda" device to be used
        NUM_EPOCHS (int): number of epochs for model to be trained
        train_loader (instance): Torch Dataloader instance for trainingset
        test_loader (instance): Torch Dataloader instance for testset
        use_l1 (bool, optional): L1 Regularization method set True to use. Defaults to False.
        scheduler (function, optional): scheduler to be used. Defaults to None.
        save_best (bool, optional): If save best model to model.pt file, paramater validation loss will be monitered
    Returns:
        (model, list): trained model and training logs
    """
    
  def train(model, device, train_loader, scheduler, optimizer, use_l1=False, lambda_l1=0.01):
    """Function to train the model
    Args:
        model (instance): torch model instance of defined model
        device (str): "cpu" or "cuda" device to be used
        train_loader (instance): Torch Dataloader instance for trainingset
        scheduler (function): scheduler to be used
        optimizer (function): optimizer to be used
        use_l1 (bool, optional): L1 Regularization method set True to use . Defaults to False.
        lambda_l1 (float, optional): Regularization parameter of L1. Defaults to 0.01.
    Returns:
        float: accuracy and loss values
    """
    
  def test(model, device, test_loader):
    """put model in eval mode and test it
    Args:
        model (instance): torch model instance of defined model
        device (str): "cpu" or "cuda" device to be used
        test_loader (instance): Torch Dataloader instance for testset
    Returns:
        float: accuracy and loss values
    """
    
  def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format
    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
```
