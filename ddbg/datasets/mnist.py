from __future__ import print_function

import torch
from torchvision import datasets, transforms

from .common import DatasetWithIDsBase

class MNISTWithIDs( DatasetWithIDsBase ):

    dataset_mean, dataset_std = 0.1307, 0.3081
    num_classes = 10
    input_size = 28
    input_channels = 1
    
    class__labels = [ str(i) for i in range(num_classes) ]
    class__1labels = class__labels


    def __init__(self):
        return super().__init__( datasets.MNIST )

    def get_train_transform( self ):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.dataset_mean,), (self.dataset_std,))
        ])
        return transform

    def get_test_transform( self ):    
        return self.get_train_transform()


'''

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index
    
def get_mnist_datasets(data_dir, train_transform, test_transform):

    train_set = MNISTWithIndices(
        data_dir,
        download=True,
        train=True,
        transform=train_transform)

    test_set = MNISTWithIndices(
        data_dir,
        download=True,
        train=False,
        transform=test_transform)

    return train_set, test_set

def get_mnist_dataloaders(data_dir, batch_size=128, num_workers=8):
    """
    mnist
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))        
    ])

    train_set, test_set = get_mnist_datasets(data_dir, train_transform, test_transform)
    num_classes = len( train_set.classes )

    train_loader = DataLoaderPrefetch(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_loader = DataLoaderPrefetch(
        test_set,
        batch_size=int(batch_size/2),
        shuffle=False,
        num_workers=int(num_workers/2))

    return train_loader, test_loader, num_classes


'''
