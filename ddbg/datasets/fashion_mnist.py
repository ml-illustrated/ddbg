from __future__ import print_function

from torchvision import datasets, transforms

from .common import DatasetWithIDsBase

class FashionMNISTWithIDs( DatasetWithIDsBase ):
    
    dataset_mean, dataset_std = 0.28604059698879553, 0.35302424451492237
    num_classes = 10
    input_size = 28
    input_channels = 1    
    
    class__labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class__1labels = ['T', 'P', 'L', 'D', 'C', 'N', 'S', 'K', 'G', 'B' ]

    def __init__(self):
        return super().__init__( datasets.FashionMNIST )

    def get_train_transform( self ):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.dataset_mean,), (self.dataset_std,))
        ])
        return transform

    def get_test_transform( self ):    
        return self.get_train_transform()


