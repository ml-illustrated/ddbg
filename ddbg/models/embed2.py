from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler


from .base_model import TrainModelBase
from .base_embed import TrainEmbeddingModelBase

class Embed2Model( TrainModelBase ):
    
    def __init__(self, opt, num_classes=10, input_channels=1, input_size=28, embed_mode=False):
        super(Embed2Model, self).__init__( opt )
        self.embed_mode = embed_mode

        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2))

        covnet_out_size = ((input_size-4)//2-4)//2

        self.fc = nn.Sequential(
            nn.Linear(64 * covnet_out_size * covnet_out_size, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2), # to be same as embedding net
        )
        self.nonlinear = nn.PReLU()        
        self.last_linear = nn.Linear(2, num_classes)

        self.softmax = nn.LogSoftmax( dim=-1 )

        self.criterion = nn.NLLLoss()
        self.criterion_no_reduction = nn.NLLLoss(reduction='none')

        self._initialize_weights()

    def get_embedding(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
        
    def forward(self, x, return_embed=False):
        embed_output = self.get_embedding(x)
        if self.embed_mode:
            return embed_output
        
        x = self.nonlinear(embed_output)
        logits = self.last_linear(x)
        x = self.softmax(logits)
        if return_embed:
            return embed_output, logits, x
        else:
            return x

    @classmethod
    def get_default_model( klass, opt, num_classes=10, input_channels=1, *args, **kwargs ):
        return klass( opt, num_classes=num_classes, input_channels=input_channels, *args, **kwargs )


class Embed2ModelEmbedOnly( TrainEmbeddingModelBase ):
    
    def __init__(self, opt, *args, **kwargs):
        super(Embed2ModelEmbedOnly, self).__init__( opt )
        
        self.model = Embed2Model( opt, *args, **kwargs )

    def forward(self, x):
        return self.model.get_embedding( x )

    @classmethod
    def get_default_model( klass, opt, *args, **kwargs ):
        return klass( opt, *args, **kwargs )
    
