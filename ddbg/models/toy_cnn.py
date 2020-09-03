from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import TrainModelBase

class ToyCNNModel( TrainModelBase ):
    
    def __init__(self, opt, num_classes=10, input_channels=1, input_size=28):
        super(ToyCNNModel, self).__init__( opt )
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        feature_size = ((28-2)-2)//2
        self.fc1 = nn.Linear(feature_size**2*64, 128)
        self.last_linear = nn.Linear(128, num_classes)
        self.last = nn.LogSoftmax( dim=1 )

        self.criterion = nn.NLLLoss()
        self.criterion_no_reduction = nn.NLLLoss(reduction='none')

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.last_linear(x)
        x = self.last(x)
        return x

    @classmethod
    def get_default_model( klass, opt, num_classes=10, input_channels=1, *args, **kwargs ):
        return klass( opt, num_classes=num_classes, input_channels=input_channels )
