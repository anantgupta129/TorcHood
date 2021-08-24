import time

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# Giving each folder a ID
def get_id_dictionary(path):
    """This Function will genrate Id's for all classes

    Args:
        path (sting): file path to the classes txt file

    Returns:
        dict: key-class, value-Id
    """
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict


def get_data(path):
    """This function will create a list of file path to the image and their respective labels
    and will split them in training and testing samples

    Args:
        path (sting): file path to the classes txt file 

    Returns:
        list: training and testing images and labels 
    """
    id_dict = get_id_dictionary(path)
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()

    for key, value in id_dict.items():
        #train_data += [cv2.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), cv2.COLOR_BGR2RGB) for i in range(500)]
        train_data += [path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)) for i in range(500)]
        train_labels += [value for i in range(500)]
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)
    print('finished loading data, in {} seconds'.format(time.time() - t))
    print('Samples for training: {}'.format(len(X_train)))
    print('Samples for testing: {}'.format(len(X_test)))

    return X_train, X_test, y_train, y_test

class ImagenetDataset(Dataset):
    """Pytoch class to generate data loaders for Tiny Image Net Dataset

    Args:
        Dataset (pytorch class):
    """
    def __init__(self, path, labels, transforms=None):
        """
        Args:
            path (list): list containing path of images 
            labels (list): respective labels for images 
            transforms (albumentations compose class, optional): Contains Image transformations to be applied. Defaults to None
        """
        self.transform = transforms
        self.path, self.labels = path, labels

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        """genrate data and label

        Args:
            idx (int): index of sample

        Returns:
            tensor: tansformed image and label
        """
        label = self.labels[idx]
        image = cv2.imread(self.path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
          # Apply transformations
          image = self.transform(image=image)['image']
          #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, label

    def get_class_to_id_dict(self, path, id_dict):
        """Create a dict of label to class 

        Args:
            path (string): file path to the classes txt file 
            id_dict (dict):  key-class, value-Id

        Returns:
            dict: get class of respective label
        """
        all_classes = {}
        result = {}
        for i, line in enumerate(open( path + 'words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word

        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])      
        return result     


