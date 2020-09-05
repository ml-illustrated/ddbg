# from https://github.com/adambielski/siamese-triplet

from __future__ import print_function

import os, math, logging
import numpy as np
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning

from .losses.contrastive import OnlineTripletLoss
from .metrics import AverageNonzeroTripletsMetric

ddbg_logger = logging.getLogger('ddbg')


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector( object ):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin,
        negative_selection_fn=hardest_negative,
        cpu=cpu
    )


def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin,
        negative_selection_fn=random_hard_negative,
        cpu=cpu
    )


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin,
        negative_selection_fn=lambda x: semihard_negative(x, margin),
        cpu=cpu
    )

class TrainEmbeddingModelBase( pytorch_lightning.LightningModule ):

    def __init__(self, cfg):
        super(TrainEmbeddingModelBase, self).__init__()
        self.lr = cfg.trainer.embedding.base_lr
        self.metrics = [
            AverageNonzeroTripletsMetric()
        ]

        self.checkpoint_save_path = cfg.trainer.embedding.checkpoint_path
        if self.checkpoint_save_path:
            os.makedirs( self.checkpoint_save_path, exist_ok=True )

            
        self.pretraind_base_load_path = cfg.model.checkpoint_path

        self.phase_one_epochs = cfg.trainer.embedding.phase_one_epochs

        margin = 1. # TODO: move to config
        self.criterion = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
        

    def configure_optimizers(self):
        optimizers = [ torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4) ]
        schedulers = [ torch.optim.lr_scheduler.StepLR(optimizers[0], self.phase_one_epochs, gamma=0.1, last_epoch=-1) ]

        return optimizers, schedulers


    def load_pretrained_base_model( self, epoch, checkpoint_load_path=None ):
        if not checkpoint_load_path:
            checkpoint_load_path = self.pretraind_base_load_path
        path = '%s/checkpoint.e%d.pt' % ( checkpoint_load_path, epoch )
        self.model.load_state_dict( torch.load(path) )

    def set_per_epoch_train_dataloader_fx( self, custom_fx ):
        self.per_epoch_train_dataloader_fx = custom_fx

    def train_dataloader( self ):
        # ????????????
        train_dataloader = None
        if self.per_epoch_train_dataloader_fx:
            train_dataloader, _ = self.per_epoch_train_dataloader_fx( self.current_epoch )

        return train_dataloader
        
    def on_epoch_start( self ):
        for metric in self.metrics:
            metric.reset()


    def training_step(self, batch, batch_nb):
        x, targets, _ = batch

        targets = targets if len(targets) > 0 else None
        if not type(x) in (tuple, list):
            x = (x,)
        if torch.cuda.is_available():
            x = tuple(d.cuda() for d in x)
            if targets is not None:
                targets = targets.cuda()

        outputs = self(*x)

        # print( 'model outputs: ', len(outputs), outputs[0].shape )
        
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if targets is not None:
            targets = (targets,)
            loss_inputs += targets

        # print( 'loss_inputs: ', len(loss_inputs), loss_inputs[0].shape, loss_inputs[-1].shape )
        loss_outputs = self.criterion( *loss_inputs )
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        # print( 'loss_outputs: ', len(loss_outputs), loss_outputs[0], type(loss_outputs[0]), loss_outputs[-1])
        
        for metric in self.metrics:
            metric(outputs, targets, loss_outputs)

        tensorboard_logs = {'train_loss': loss}
        for metric in self.metrics:
            tensorboard_logs[ metric.name() ] = metric.value()
            
        return {'loss': loss, 'log': tensorboard_logs}

    
    def training_epoch_end(self, outputs):

        losses_mean = np.array( [ i['loss'].detach().cpu().numpy() for i in outputs ] ).mean()
        msg = 'Train Epoch %d Loss: %g' % ( self.current_epoch, losses_mean )
        
        metrics_dict = { 'loss_mean' : losses_mean }
        for metric in self.metrics:
            msg += '\t%s: %0.2f' % ( metric.name(), metric.value() )
            metrics_dict[ metric.name() ] = metric.value()

        ddbg_logger.info( msg )
            
        if self.checkpoint_save_path:
            path = '%s/checkpoint.e%d.pt' % ( self.checkpoint_save_path, self.current_epoch )
            torch.save( self.state_dict(), path)

        return metrics_dict

    def load_pretrained( self, epoch, checkpoint_load_path=None ):
        if checkpoint_load_path==None:
            checkpoint_load_path = self.checkpoint_save_path
        path = '%s/checkpoint.e%d.pt' % ( checkpoint_load_path, epoch )
        ddbg_logger.debug( 'load pretrained embed %s' % path )
        self.load_state_dict( torch.load(path) )
