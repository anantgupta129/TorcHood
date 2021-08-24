import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def convert_image_np(inp, mean, std):
    """Convert a Tensor to numpy image.

    Args:
        inp (tensor): Tensor image
        mean(np array): numpy array of mean of dataset
        std(np array): numpy array of standard deviation of dataset

    Returns:
        np array: a numpay image
    """

    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def plot_data(data, rows, cols, lower_value, upper_value):
    """Randomly plot the images from the dataset for vizualization

    Args:
        data (instance): torch instance for data loader
        rows (int): number of rows in the plot
        cols (int): number of cols in the plot
        lower_value (int): lower value of the dataset for plotting in a particular interval. 0 for starting index
        upper_value (int): upper value for plotting in a particular interval. len of dataset for last index index
    """
    figure = plt.figure(figsize=(cols*2,rows*3))
    for i in range(1, cols*rows + 1):
        k = np.random.randint(lower_value,upper_value)
        figure.add_subplot(rows, cols, i) # adding sub plot

        img, label = data.dataset[k]
        
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Class: {label}')

    plt.tight_layout()
    plt.show()


def plot_aug(aug_dict, data, ncol=6):
    """Vizualize the image for the augmentations to be applied over dataset

    Args:
        aug_dict (dict): dictionary key as name of augmentation to applied (str) and 
                                    value as albumentations aug function
        data (instance): torch instance for data loader
        ncol (int, optional): number of cols in the plot. Defaults to 6.
    """
    nrow = len(aug_dict)

    fig, axes = plt.subplots(ncol, nrow, figsize=( 3*nrow, 15), squeeze=False)
    for i, (key, aug) in enumerate(aug_dict.items()):
        for j in range(ncol):
            ax = axes[j,i]
            if j == 0:
                ax.text(0.5, 0.5, key, horizontalalignment='center', verticalalignment='center', fontsize=15)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.axis('off')
            else:
                image, label = data.dataset[j-1]
                if aug is not None:
                    transform = A.Compose([aug])
                    image = np.array(image)
                    image = transform(image=image)['image']
                
                ax.imshow(image)
                #ax.set_title(f'{data.classes[label]}')
                ax.axis('off')

    plt.tight_layout()
    plt.show()
    

def plot_misclassified(model, test_loader, classes, device, dataset_mean, dataset_std, no_misclf=20, plot_size=(4,5), return_misclf=False):
    """Plot the images are wrongly clossified by model

    Args:
        model (instance): torch instance of defined model (pre trained)
        test_loader (instace): torch data loader of testing set
        classes (dict or list): classes in the dataset
                if dict:
                    key - class id
                    value - as class name
                elif list:
                    index of list correspond to class id and name
        device (str): 'cpu' or 'cuda' device to be used
        dataset_mean (tensor or np array): mean of dataset
        dataset_std (tensor or np array): std of dataset
        no_misclf (int, optional): number of misclassified images to plot. Defaults to 20.
        plot_size (tuple): tuple containing size of plot as rows, columns. Defaults to (4,5)
        return_misclf (bool, optional): True to return the misclassified images. Defaults to False.

    Returns:
        list: list containing misclassified images as np array if return_misclf True
    """
    count = 0
    k = 0
    misclf = list()
  
    while count<no_misclf:
        img, label = test_loader.dataset[k]
        pred = model(img.unsqueeze(0).to(device)) # Prediction
        # pred = model(img.unsqueeze(0).to(device)) # Prediction
        pred = pred.argmax().item()

        k += 1
        if pred!=label:
            img = convert_image_np(
                img, dataset_mean, dataset_std)
            misclf.append((img, label, pred))
            count += 1
    
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*3,rows*3))

    for i in range(1, cols * rows + 1):
        img, label, pred = misclf[i-1]

        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img, cmap="gray") # showing the plot

    plt.tight_layout()
    plt.show()
    
    if return_misclf:
        return misclf


def plot_learning_curves(history=None, from_txt=False, plot_lr_trend=False):
    """Plot Test & Train Learning Curves of model

    Args:
        history (tuple, optional): tuple of list contraing (training_acc, training_loss, testing_acc, testing_loss)
                            Note- in specific order only. Defaults to None. if information is dumped to txt file
        from_txt (bool or list, optional): List of path to (training_acc, training_loss, testing_acc, testing_loss) txt files
                            Note- give path in specific order only. Defaults to False.
        plot_lr_trend (bool or list or str, optional): List if plot learning curve trend form list 
                                                       str- plot from txt file, path to txt file. 
                                                       Defaults to False. (dont plot)
    """
    if from_txt:
        history = []
        for path in from_txt:
            history.append([float(i) for i in open(path).read().strip().split("\n")])
    

    fig, axs = plt.subplots(1,2,figsize=(16,7))
    axs[0].set_title('LOSS')
    axs[0].plot(history[1], label='Train')
    axs[0].plot(history[3], label='Test')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Accuracy')
    axs[1].plot(history[0], label='Train')
    axs[1].plot(history[2], label='Test')
    axs[1].legend()
    axs[1].grid()

    plt.show()

    if plot_lr_trend:
        if from_txt:
            lr_trend = [float(i) for i in open(plot_lr_trend).read().strip().split("\n")]
        else:
            lr_trend = plot_lr_trend
        
        plt.plot(lr_trend)
        plt.title('Learning Rate Change During Training')
        plt.xlabel('Iteration')
        plt.ylabel('LR')
        plt.grid()
        plt.show()


def visualize_stn(model, test_loader, device, mean, std):
    """Below are Vizulation codes for Spatial Transformer

    We want to visualize the output of the spatial transformers layer
    after the training, we visualize a batch of input images and
    the corresponding transformed batch using STN.

    Args:
        model (instance): pytorch instance of model
        test_loader (instace): pytorch instance of test data loader
        device (str): device to de used
        mean(np array): numpy array of mean of dataset
        std(np array): numpy array of standard deviation of dataset
    """
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            make_grid(input_tensor), mean, std)

        out_grid = convert_image_np(
            make_grid(transformed_input_tensor), mean, std)

        # Plot the results side-by-side
        f, axs = plt.subplots(1, 2, figsize=(len(data)/10*3, len(data)/10*5))
        axs[0].imshow(in_grid)
        axs[0].set_title('Dataset Images')

        axs[1].imshow(out_grid)
        axs[1].set_title('Transformed Images')
    
    plt.ioff()
    plt.show()



