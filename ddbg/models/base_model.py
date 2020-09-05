from __future__ import print_function
import os, math, logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning
from pytorch_lightning.trainer.trainer import _PatchDataLoader

from torch.optim.lr_scheduler import _LRScheduler

from .metrics import AccumulatedAccuracyMetric

ddbg_logger = logging.getLogger('ddbg')


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, lr, lr_decay_rate, max_epochs, last_epoch=-1):
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.max_epochs = max_epochs
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.lr
        epoch = self.last_epoch+1
        eta_min = lr * (self.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / self.max_epochs)) / 2
        ddbg_logger.debug( 'cos lr %g' % lr )

        return [ lr for param_group in self.optimizer.param_groups ]


class TrainModelBase( pytorch_lightning.LightningModule ):

    def __init__(self, cfg):
        super(TrainModelBase, self).__init__()

        #### training parameters
        self.optimizer_name = cfg.trainer.base.optimizer.name.lower()
        self.lr = cfg.trainer.base.base_lr
        self.lr_steps = None
        if cfg.trainer.base.lr_schedule.steps:
            self.lr_steps = [ s for s in cfg.trainer.base.lr_schedule.steps if s != -1 ]
            # self.lr_steps = [ int(i) for i in lr_steps.split(',') ]

        self.max_epochs = cfg.trainer.base.epochs
        self.cosine_schedule = cfg.trainer.base.lr_schedule.cosine
        self.lr_decay_rate = cfg.trainer.base.lr_decay_rate
        self.momentum = cfg.trainer.base.optimizer.momentum
        self.weight_decay = cfg.trainer.base.optimizer.weight_decay
            
            
        self.checkpoint_save_path = cfg.model.checkpoint_path
        if self.checkpoint_save_path:
            os.makedirs( self.checkpoint_save_path, exist_ok=True )

        self.metric = AccumulatedAccuracyMetric()
        self.per_sample_weights = None

        self.per_epoch_train_dataloader_fx = None

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr
            )
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError( 'unsupported optimizer %s' % self.optimizer_name )
        
        optimizers = [optimizer]
        schedulers = []

        if self.cosine_schedule:
            schedulers.append( CosineLR( optimizers[0], self.lr, self.lr_decay_rate, self.max_epochs ) )
        elif self.lr_steps:
            schedulers.append( torch.optim.lr_scheduler.MultiStepLR(optimizers[0], self.lr_steps, gamma=0.1, last_epoch=-1) )

        return optimizers, schedulers


    def set_sample_weights(self, sample_weights):
        self.per_sample_weights = sample_weights

    def set_per_epoch_train_dataloader_fx( self, custom_fx ):
        self.per_epoch_train_dataloader_fx = custom_fx

    def train_dataloader( self ):

        train_dataloader = None
        if self.per_epoch_train_dataloader_fx:
            train_dataloader, _ = self.per_epoch_train_dataloader_fx( self.current_epoch )

        return train_dataloader
        
    def on_epoch_start( self ):
        self.metric.reset()

        
    def training_step(self, batch, batch_nb):
        x, targets, _ = batch
        logits = self(x)
        loss = self.criterion( logits, targets )

        self.metric( [logits], [targets], loss)
        
        tensorboard_logs = {'train_loss': loss, 'train_acc': self.metric.value() }
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        losses_sum = np.array( [ i['loss'].detach().cpu().numpy() for i in outputs ] ).sum()
        ddbg_logger.info( 'Train epoch %d Loss: %g Accuracy: %0.2f%%' % ( self.current_epoch, losses_sum, self.metric.value() ) )
        
        if self.checkpoint_save_path:
            path = os.path.join( self.checkpoint_save_path, 'checkpoint.e%d.pt' % self.current_epoch )
            torch.save( self.state_dict(), path)
            
        return { 'test_acc': self.metric.value(), 'loss_sum' : losses_sum }

    '''
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.criterion(self(x), y)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    '''

    def test_step(self, batch, batch_idx):
        x, targets, x_idxes = batch
        logits = self(x)
        loss = self.criterion( logits, targets )

        self.metric( [logits], [targets], loss)
        
        tensorboard_logs = {'test_loss': loss, 'test_acc': self.metric.value() }
        return {'test_loss': loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        ddbg_logger.info( 'Test Accuracy: %0.2f%%' % ( self.metric.value() ) )
        
        return { 'test_acc': self.metric.value() }
    

    @classmethod
    def load_training_checkpoints( klass, cfg, epochs_to_load, *args, **kwargs ):
        checkpoint_load_path = cfg.model.checkpoint_path
        models = []
        for epoch in epochs_to_load:
            model = klass.get_default_model( cfg, *args, **kwargs )
            path = os.path.join( checkpoint_load_path, 'checkpoint.e%d.pt' % epoch )
            model.load_state_dict( torch.load(path) )
            models.append( model )
        return models

    def load_pretrained( self, checkpoint_load_path, epoch ):
        # checkpoint_load_path = cfg.model.checkpoint_path        
        path = os.path.join( checkpoint_load_path, 'checkpoint.e%d.pt' % epoch )
        self.load_state_dict( torch.load(path) )
    

    def _initialize_weights( self, mode='kaiming' ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                elif mode == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=1)
                elif mode == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)


