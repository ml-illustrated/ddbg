from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler


from .base_model import TrainModelBase
from .base_embed import TrainEmbeddingModelBase

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 remove_first_relu,
                 add_last_bn,
                 preact=False):
        super().__init__()

        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x),
                       inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        if self._add_last_bn:
            y = self.bn3(y)

        y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 remove_first_relu,
                 add_last_bn,
                 preact=False):
        super().__init__()

        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact

        bottleneck_channels = out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        if add_last_bn:
            self.bn4 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x),
                       inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)

        if self._add_last_bn:
            y = self.bn4(y)

        y += self.shortcut(x)
        return y

class ResnetPreactModel( TrainModelBase ):

    def __init__(self, opt, num_classes=10, input_channels=3, input_size=32, depth=20, embed_mode=False):
        super().__init__( opt )

        '''
        model_config = config.model.resnet_preact
        initial_channels = model_config.initial_channels
        self._remove_first_relu = model_config.remove_first_relu
        self._add_last_bn = model_config.add_last_bn
        block_type = model_config.block_type
        depth = model_config.depth
        preact_stage = model_config.preact_stage
        '''
        initial_channels = 16
        self._remove_first_relu = False
        self._add_last_bn = False
        block_type = 'basic'
        preact_stage = [True, True, True]
        self.embed_mode = embed_mode

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth

        n_channels = [
            initial_channels,
            initial_channels * 2 * block.expansion,
            initial_channels * 4 * block.expansion,
        ]

        self.conv = nn.Conv2d(input_channels,
                              n_channels[0],
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1,
                              bias=False)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[0],
                                       n_blocks_per_stage,
                                       block,
                                       stride=1,
                                       preact=preact_stage[0])
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2,
                                       preact=preact_stage[1])
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2,
                                       preact=preact_stage[2])
        self.bn = nn.BatchNorm2d(n_channels[2])

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, input_channels, input_size, input_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        # print( 'self.feature_size: ', self.feature_size ) # 64
        # self.fc = nn.Linear(self.feature_size, num_classes)
        self.last_linear = nn.Linear(self.feature_size, num_classes)

        # initialize weights
        # initializer = create_initializer(config.model.init_mode)
        # self.apply(initializer)

        '''
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2), # to be same as embedding net
        )
        self.nonlinear = nn.ReLU()        
        self.last_linear = nn.Linear(2, num_classes)
        '''
        self.softmax = nn.LogSoftmax( dim=-1 )

        self.criterion = nn.NLLLoss()
        self.criterion_no_reduction = nn.NLLLoss(reduction='none')

        self._initialize_weights()
        

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride,
                    preact):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    block(in_channels,
                          out_channels,
                          stride=stride,
                          remove_first_relu=self._remove_first_relu,
                          add_last_bn=self._add_last_bn,
                          preact=preact))
            else:
                stage.add_module(
                    block_name,
                    block(out_channels,
                          out_channels,
                          stride=1,
                          remove_first_relu=self._remove_first_relu,
                          add_last_bn=self._add_last_bn,
                          preact=False))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x),
                   inplace=True)  # apply BN and ReLU before average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def get_embedding(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x, return_embed=False):
        embed_output = self.get_embedding(x)
        if self.embed_mode:
            return embed_output

        logits = self.last_linear(embed_output)
        x = self.softmax(logits)
        if return_embed:
            return embed_output, logits, x
        else:
            return x

    @classmethod
    def get_default_model( klass, opt, num_classes=10, input_channels=3, depth=20, *args, **kwargs ):
        model = klass( opt, num_classes=num_classes, input_channels=input_channels, depth=depth, *args, **kwargs )
        # print( model )
        return model
    

class ResnetPreact50Model( ResnetPreactModel ):
    @classmethod
    def get_default_model( klass, opt, num_classes=10, input_channels=3, *args, **kwargs ):
        model = klass( opt, num_classes=num_classes, input_channels=input_channels, depth=50, *args, **kwargs )
        # print( model )
        return model




class ResnetPreact50ModelEmbedOnly( TrainEmbeddingModelBase ):
    
    def __init__(self, opt, *args, **kwargs):
        super(ResnetPreact50ModelEmbedOnly, self).__init__( opt )
        
        self.model = ResnetPreact50Model.get_default_model( opt, *args, **kwargs )

    def forward(self, x):
        return self.model.get_embedding( x )

    @classmethod
    def get_default_model( klass, opt, *args, **kwargs ):
        return klass( opt, *args, **kwargs )
