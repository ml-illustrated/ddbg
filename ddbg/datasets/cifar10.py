from __future__ import print_function

import numpy as np
from torchvision import datasets, transforms

from .common import DatasetWithIDsBase

class CIFAR10WithIDs( DatasetWithIDsBase ):
    
    dataset_mean = np.array([0.4914, 0.4822, 0.4465])
    dataset_std = np.array([0.2470, 0.2435, 0.2616])

    num_classes = 10
    input_size = 32
    input_channels = 3
    
    class__labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    class__1labels = ['P', 'A', 'B', 'C', 'D', 'G', 'F', 'H', 'S', 'T' ]

    def __init__(self):
        return super().__init__( datasets.CIFAR10 )

    def get_train_transform( self ):
        transform = transforms.Compose([
            transforms.RandomCrop(32,
                                  padding=4,
                                  fill=0,
                                  padding_mode='constant'),
            transforms.RandomHorizontalFlip(0.5),            
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_mean, self.dataset_std)
        ])
        return transform

    def get_test_transform( self ):    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_mean, self.dataset_std)
        ])
        return transform


