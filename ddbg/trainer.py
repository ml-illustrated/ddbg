import psutil
import os, logging
import numpy as np
import torch
import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger

from ddbg.datasets.common import BalancedBatchSampler
from ddbg.self_influence import SelfInfluenceResults

ddbg_logger = logging.getLogger('ddbg')

class ModelTrainer( object ):

    def __init__( self, config, dataset, model ):
        self.dataset = dataset
        self.model = model

        # pull out relevant settings from config
        self.dataset_dir = config.dataset.dataset_path
        self.gpus = config.trainer.gpus
        self.max_epochs = config.trainer.base.epochs
        self.batch_size = config.trainer.base.batch_size
        self.num_workers = config.trainer.num_workers        

        self.log_dir = config.project.log_dir        
        self.project_name = config.project.name

        self.model.set_per_epoch_train_dataloader_fx( self.get_per_epoch_dataloaders )

    def train( self ):
        train_loader, _ = self.get_per_epoch_dataloaders( 0, download=True ) # preload
        refresh_rate = int( len( train_loader ) * 0.05 ) # refresh every 5%
        del train_loader
        
        tb_logger = TensorBoardLogger( save_dir=self.log_dir, name='%s_base' % self.project_name )
        trainer = pytorch_lightning.Trainer(
            max_epochs = self.max_epochs,
            gpus = self.gpus,
            progress_bar_refresh_rate = refresh_rate,
            reload_dataloaders_every_epoch = True,
            logger = tb_logger,
            checkpoint_callback = False, # disable since saving every epoch already
        )

        # begin training
        trainer.fit( self.model )


    def eval_base_model_metric( self ):
        _, test_loader = self.get_per_epoch_dataloaders( -1 )
        
        self.model.metric.reset()
        for i, test_batch in enumerate( test_loader ):
            _ = self.model.test_step( test_batch, i )

        model_metric = self.model.metric.value()
        ddbg_logger.info( 'Base model Test_accuracy: %0.2f%%' % ( model_metric ) )
        return model_metric
        


    def get_per_epoch_dataloaders( self, epoch_num, download=False ):
        batch_size = self.batch_size
        train_loader, test_loader = self.dataset.get_dataloaders(
            data_dir=self.dataset_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            download=download,
        )
        return train_loader, test_loader
        


class EmbeddingTrainer( object ):

    def __init__(
            self,
            config,
            dataset,
            model,
            self_influence_results: SelfInfluenceResults = None,
    ):
        
        self.dataset = dataset
        self.model = model

        # pull out relevant settings from config
        self.dataset_dir = config.dataset.dataset_path
        self.gpus = config.trainer.gpus
        self.phase_one_epochs = config.trainer.embedding.phase_one_epochs
        self.max_epochs = config.trainer.embedding.epochs
        self.samples_per_class = config.trainer.embedding.samples_per_class
        self.self_influence_percentile = config.trainer.embedding.self_influence_percentile
        self.pretrained_base_epoch = config.trainer.embedding.pretrained_base_epoch
        self.num_workers = config.trainer.num_workers        

        self.log_dir = config.project.log_dir        
        self.project_name = config.project.name

        assert self.max_epochs >= self.phase_one_epochs, 'max_epochs must be equal or greater than phase_one_epochs'

        if self.self_influence_percentile:
            self_influence_scores = self_influence_results.self_influence_scores
            self.self_influence_sorted_dataset_indices = np.argsort( self_influence_scores )
        else:
            self.self_influence_sorted_dataset_indices = None
            
        self.model.set_per_epoch_train_dataloader_fx( self.get_per_epoch_dataloaders )

        if self.pretrained_base_epoch:
            self.model.load_pretrained_base_model( self.pretrained_base_epoch )
            ddbg_logger.info( 'loaded base model epoch %d' % self.pretrained_base_epoch )
        
    def train( self ):
        train_loader, _ = self.get_per_epoch_dataloaders( 0, download=True ) # preload        
        refresh_rate = int( len( train_loader ) * 0.05 ) # refresh every 5%
        del train_loader
        
        tb_logger = TensorBoardLogger( save_dir=self.log_dir, name='%s_embed' % self.project_name )
        trainer = pytorch_lightning.Trainer(
            max_epochs = self.max_epochs,
            gpus = self.gpus,
            progress_bar_refresh_rate = refresh_rate,
            reload_dataloaders_every_epoch = True,
            logger = tb_logger,
            checkpoint_callback = False, # disable checkpoints
        )

        # begin training
        trainer.fit( self.model )
    

    def get_per_epoch_dataloaders( self, epoch_num, download=False ):
        percentile = self.self_influence_percentile
        if percentile:
            if epoch_num+1 >= self.phase_one_epochs:
                per_epoch_percentile = (1-percentile) / (self.max_epochs - self.phase_one_epochs)
                percentile = min( 1, percentile+per_epoch_percentile*((epoch_num+1)-(self.phase_one_epochs-1)) )
                
            cutoff = int(len( self.self_influence_sorted_dataset_indices )*percentile)
            train_subset_indicies = self.self_influence_sorted_dataset_indices[:cutoff]
            ddbg_logger.debug( 'train_subset_indicies epoch %s len: %s' % ( epoch_num+1, len( train_subset_indicies ) ) )
        else:
            train_subset_indicies = None

        train_dataset, test_dataset = self.dataset.get_datasets(
            self.dataset_dir,
            self.dataset.get_train_transform(),
            self.dataset.get_test_transform(),
            train_subset_indicies=train_subset_indicies,
            download=download,
        )

        n_classes = self.dataset.num_classes
        train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=n_classes, n_samples=self.samples_per_class)
        test_batch_sampler = BalancedBatchSampler(test_dataset.targets, n_classes=n_classes, n_samples=self.samples_per_class)

        kwargs = {'num_workers': self.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
        online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

        return online_train_loader, online_test_loader
        
        
