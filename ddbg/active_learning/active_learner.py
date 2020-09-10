import os, logging, psutil
from typing import Tuple

import numpy as np
import torch

from ddbg import DatasetDebugger
# from ddbg.self_influence import DatasetSelfInfluence, SelfInfluenceResults

ddbg_logger = logging.getLogger('ddbg')

class ActiveLearner( object ):

    def __init__(
            self,
            project_config,
    ):

        self.cfg = project_config
        self.dataset = None

        self.model_name = self.cfg.model.arch.lower()
        self.init_num_samples = self.cfg.active_learning.init_num_samples
        self.per_cycle_train_epochs = self.cfg.active_learning.per_cycle_train_epochs
        self.per_cycle_num_samples = self.cfg.active_learning.per_cycle_num_samples
        self.total_num_cycles = self.cfg.active_learning.total_num_cycles
        self.unlabeled_rank_method = self.cfg.active_learning.unlabeled_rank_method
        self.unlabeled_similary_measure = self.cfg.active_learning.unlabeled_similary_measure


    def run_active_learning_simulation( self ):

        ddbg_project = DatasetDebugger( self.cfg )
        train_loader, _ = ddbg_project._get_dataset_for_data_influence()
        
        num_samples = len(train_loader.dataset)
        np.random.seed( self.init_num_samples ) # keep init subset the same
        indices = np.random.permutation( num_samples )
        initial_train_subset_indices = indices[ :self.init_num_samples ]
        remain_indices = indices[ self.init_num_samples: ]

        cycle_num = 0
        per_cycle_metrics = []
        train_subset_indices = initial_train_subset_indices
        model = None
        for cycle_num in range( self.total_num_cycles ):
            model, metric = self.run_train_cycle( cycle_num, train_subset_indices, model=model )
            per_cycle_metrics.append( metric )
            train_subset_indices, remain_indices = self.get_next_cycle_indices( train_subset_indices, remain_indices, model, train_loader )


    def run_train_cycle( self, cycle_num, train_subset_indices, model ):
        train_cfg = self.cfg.clone()
        # train_cfg.max_epochs = self.per_cycle_train_epochs
        train_cfg.log_dir = '%s_cycle%s' % ( train_cfg.project.log_dir, cycle_num )
        train_cfg.checkpoint_save_path = '%s_cycle%s' % ( train_cfg.model.checkpoint_path, cycle_num )

        ddbg_project = DatasetDebugger( train_cfg )
        ddbg_logger.info( 'Train cycle %d subset len %d' % ( cycle_num, train_subset_indices.shape[0] ) )
        model, metric = ddbg_project.train_base_model( model=model, train_subset_indices=train_subset_indices )
        return model, metric

    def get_next_cycle_indices( self, train_subset_indices, remain_indices, model, train_loader ):

        if self.unlabeled_rank_method == 'random':
            next_batch_indices = remain_indices[ :self.per_cycle_num_samples ]
            remain_indices = remain_indices[ self.per_cycle_num_samples: ]
        elif self.unlabeled_rank_method == 'min_logits':
            ddbg_logger.info( 'Calculating logits of remaining datapoints..' )
            _, logits = self.model__embed_logits( model, train_loader )
            logits_of_remain = logits[ remain_indices ]
            logits_max = np.amax( logits_of_remain, axis=1)
            logit_ranked_indicies = remain_indices[ np.argsort( logits_max ) ]

            next_batch_indices = logit_ranked_indicies[ :self.per_cycle_num_samples ]
            remain_indices = logit_ranked_indicies[ self.per_cycle_num_samples: ]
            
        else:
            raise NotImplementedError( 'unknown rank method %s!' % self.unlabeled_rank_method )
            
        train_subset_indices = np.concatenate( [ train_subset_indices, next_batch_indices ] )

        return train_subset_indices, remain_indices


    def model__embed_logits( self, model, data_loader ):
        if torch.cuda.is_available():
            model = model.cuda()

        all_embed_activations = []
        all_logits = []
        for inputs, _, input_indexes in data_loader:
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
        
            with torch.no_grad():
                embed_activations, logits, _ = model(inputs, return_embed=True)
        
                all_embed_activations.append( embed_activations.detach().cpu().numpy().copy()  )
                all_logits.append( logits.detach().cpu().numpy().copy()  )
                del inputs
                del embed_activations
                del logits

        return np.concatenate( all_embed_activations ), np.concatenate( all_logits )

        
