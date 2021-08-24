import torch

def get_mean_std(loader):
    """Calculate mean and standard deviation of the dataset

    Args:
        loader (instance): torch instance for data loader

    Returns:
        tensor: mean and std of data
    """
    channel_sum, channel_squared_sum,  num_batches = 0,0,0
    
    for img,_ in loader:
        channel_sum += torch.mean(img/255., dim=[0,1,2])
        channel_squared_sum += torch.mean((img/255.)**2, dim=[0,1,2])
        num_batches += 1
        
    mean = channel_sum / num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print("The mean of dataset : ", mean)
    print("The std of dataset : ", std)
    return mean,std

