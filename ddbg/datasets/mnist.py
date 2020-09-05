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

