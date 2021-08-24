from torch_lr_finder import LRFinder


def find_lr(net, optimizer, criterion, train_loader):
    """Find learning rate for using One Cyclic LRFinder

    Args:
        net (instace): torch instace of defined model
        optimizer (instance): optimizer to be used
        criterion (instance): criterion to be used for calculating loss
        train_loader (instance): torch dataloader instace for trainig set
    """
    lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot()
    lr_finder.reset()


