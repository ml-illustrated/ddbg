from typing import Tuple

import os, logging
import numpy as np
import joblib
import psutil

import torch
import torch.nn as nn

from tqdm.auto import tqdm

ddbg_logger = logging.getLogger('ddbg')

class ProponentOpponentsResults( object ):

    def __init__(
            self,
            dataset_loss_grads: np.ndarray,
            dataset_embed_activations: np.ndarray,
            dataset_target_labels: np.ndarray,
            final_predicted_labels: np.ndarray,
            final_predicted_prob: np.ndarray,
            dataset_top_proponents = None,
            dataset_top_opponents = None,
            
    ):
        self.dataset_loss_grads = dataset_loss_grads
        self.dataset_embed_activations = dataset_embed_activations
        self.dataset_target_labels = dataset_target_labels
        self.final_predicted_labels = final_predicted_labels
        self.final_predicted_prob = final_predicted_prob
        self.dataset_top_proponents = dataset_top_proponents
        self.dataset_top_opponents = dataset_top_opponents

    def set_top_prop_oppos(
            self,
            dataset_top_proponents: np.ndarray,
            dataset_top_opponents: np.ndarray,
    ):
        self.dataset_top_proponents = dataset_top_proponents
        self.dataset_top_opponents = dataset_top_opponents
                

    def save( self, output_path ):
        out_dir = os.path.split( output_path )[0]
        os.makedirs( out_dir, exist_ok=True )

        with open( output_path, 'wb' ) as fp:
            # metadata?
            np.save( fp, self.dataset_loss_grads )
            np.save( fp, self.dataset_embed_activations )
            np.save( fp, self.dataset_target_labels )
            np.save( fp, self.final_predicted_labels )
            np.save( fp, self.final_predicted_prob )
            np.save( fp, self.dataset_top_proponents )
            np.save( fp, self.dataset_top_opponents )

    @classmethod
    def load( klass, load_path ):
            
        with open( load_path, 'rb' ) as fp:
            dataset_loss_grads = np.load( fp )
            dataset_embed_activations = np.load( fp )
            dataset_target_labels = np.load( fp )
            final_predicted_labels = np.load( fp )
            final_predicted_prob = np.load( fp )
            dataset_top_proponents = np.load( fp )
            dataset_top_opponents  = np.load( fp )

        return klass(
            dataset_loss_grads = dataset_loss_grads,
            dataset_embed_activations = dataset_embed_activations,
            dataset_target_labels = dataset_target_labels,
            final_predicted_labels = final_predicted_labels,
            final_predicted_prob = final_predicted_prob,
            dataset_top_proponents = dataset_top_proponents,
            dataset_top_opponents  = dataset_top_opponents,
        )        


class ProgressParallel(joblib.Parallel):
    def __call__(self, total_calls, *args, **kwargs):
        with tqdm( total=total_calls ) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        # self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        if ( self._pbar.n % 100 ) == 0:
            self._pbar.refresh()

class DatasetProponentOpponents( object ):

    def __init__(
            self,
            models,
            data_loader,
    ):
        self.models = models
        self.data_loader = data_loader

    def calc_proponent_opponent_results(
            self,
            precalc_all_prop_oppos: bool = True,
            top_k: int = 5
    ) -> ProponentOpponentsResults:
        
        prop_oppo_results = self.gen_trackin_gradients()

        if precalc_all_prop_oppos:
            ddbg_logger.info( 'Precomputing dataset proponents & opponents' )
            
            dataset_top_proponents, dataset_top_opponents = self.precompute_dataset__top_proponents_opponents(
                prop_oppo_results,
                top_k=top_k,
            )

            prop_oppo_results.set_top_prop_oppos(
                dataset_top_proponents,
                dataset_top_opponents
            )
        
        return prop_oppo_results

    def gen_trackin_gradients(
            self,
    ) -> ProponentOpponentsResults:

        n_batches = len( self.data_loader ) 
        progress_bar = tqdm(
            desc='Prop & Oppos',
            total=n_batches,
            initial=0,
        )
        
        all_loss_grads = []
        all_targets = []
        all_embed_activations = []
        all_last_output_probs = []
        all_last_predicted_labels = []
        num_classes = len( self.data_loader.dataset.classes )
        for idx, (inputs, targets, input_indexes) in enumerate( self.data_loader ):
            # inputs.shape:  torch.Size([64, 3, 32, 32])
            # targets.shape: torch.Size([64])
            all_targets.append( targets )
        
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
        
            prediction_losses_outputs = self._batch__prediction_losses( inputs, targets, num_classes )
            batch_loss_grads, batch_embed_activations, last_output_probs, last_predicted_labels = prediction_losses_outputs
        
            all_loss_grads.append( batch_loss_grads.detach().cpu().numpy().copy() )
            all_embed_activations.append( batch_embed_activations.detach().cpu().numpy().copy() )
        
            all_last_output_probs.append( last_output_probs.detach().cpu().numpy().copy() )
            all_last_predicted_labels.append( last_predicted_labels.detach().cpu().numpy().copy() )

            del inputs
            del targets
            del last_predicted_labels
            del batch_loss_grads
            del batch_embed_activations
            del last_output_probs
            # update progress
            progress_bar.update(1)
        
        results = ProponentOpponentsResults( 
            dataset_loss_grads = np.concatenate( all_loss_grads ),
            dataset_embed_activations = np.concatenate( all_embed_activations ),
            dataset_target_labels = np.concatenate( all_targets ),
            final_predicted_labels = np.concatenate( all_last_output_probs ),
            final_predicted_prob = np.concatenate( all_last_predicted_labels ),
        )
        return results
            
    
    def _batch__prediction_losses(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            num_classes: int,
    ) -> Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]:
    
        batch_loss_grads = []
        batch_embed_activations = []
        for model in self.models:
            loss_grads, embed_activations, cls_probs = self._model__prediction_losses( model, inputs, targets, num_classes )
            batch_loss_grads.append( loss_grads )
            batch_embed_activations.append( embed_activations )

        # use last model's outputs
        last_output_probs, last_predicted_labels = torch.topk( cls_probs, 1 )

        batch_loss_grads = torch.stack( batch_loss_grads, axis=-1 )
        batch_embed_activations = torch.stack( batch_embed_activations, axis=-1 )
        
        return batch_loss_grads, batch_embed_activations, last_output_probs, last_predicted_labels


    def _model__prediction_losses(
            self,
            model,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            num_classes: int,
    ) -> Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ]:

        num_inputs = inputs.shape[0]

        with torch.no_grad():
            embed_activations, logits, _ = model(inputs, return_embed=True) # ( (64,2), [64, 10] ) # for embed2 on mnist
            cls_output = nn.functional.softmax( logits, dim=-1 )
            one_hot = torch.zeros(num_inputs, num_classes, device=logits.device)
            one_hot[ torch.arange(num_inputs), targets] = 1.
            loss_grads = one_hot - cls_output

        return loss_grads, embed_activations, cls_output


        
    def datapoint_output__top_prop_oppo(
            self,
            target_class: int,
            loss_grad: np.ndarray,
            activation: np.ndarray,
            prop_oppo_results: ProponentOpponentsResults,
            top_k: int=5,
    ) -> Tuple[ np.ndarray, np.ndarray ]:

        train_loss_grads = prop_oppo_results.dataset_loss_grads
        train_activations = prop_oppo_results.dataset_embed_activations
        target_labels = prop_oppo_results.dataset_target_labels
        # last_predicted_labels = prop_oppo_results.final_predicted_labels
        
        loss_grad_similarity = np.sum( train_loss_grads * loss_grad, axis=(1,2) ) # -> (60000)
        # find the -embeds w/ min the absolute differeces to input embed
        loss_grads_diff = np.abs( train_loss_grads + loss_grad ) # smaller == more similar in opposites
        # penalize similar -embeds with larget diff of target class
        loss_grad_dissimilarity = np.clip( np.sum( np.sum( loss_grads_diff, axis=1 ) * (1+loss_grads_diff[ :,target_class,: ]), axis=1 ), 1.0, None )
    
        # penalize similarity in embeddings with large differences
        # loss_grad_similarity_normed = loss_grad_similarity / ( loss_grad_dissimilarity ** 2 ) # favors larger target class confusion
        loss_grad_similarity_normed = loss_grad_similarity / loss_grad_dissimilarity

        activation_similarity = np.sum( train_activations * activation, axis=(1,2) ) # -> (60000,)
        # zero out any activation that's negative
        activation_similarity[ activation_similarity < 0 ] = 0.0
        combined_scores = loss_grad_similarity_normed * activation_similarity
    
        opponents = []
        proponents = []
        indices = np.argsort(combined_scores)
        for i in range(top_k):
            index = indices[-i-1]
            proponents.append( np.array( [
                float( index ),
                combined_scores[index],
                loss_grad_similarity[index],
                activation_similarity[index],
            ] ) )
            index = indices[i]
            # check for when opponents are not 
            if loss_grad_similarity[index] > 0: continue
            opponents.append( np.array( [
                float( index ),
                combined_scores[index],
                loss_grad_similarity[index],
                activation_similarity[index],
            ] ) )

        return np.array( proponents ), np.array( opponents )


    def precompute_dataset__top_proponents_opponents(
            self,
            prop_oppo_results: ProponentOpponentsResults,
            top_k: int = 5,
    ) -> Tuple[ np.ndarray, np.ndarray ]:

        train_loss_grads = prop_oppo_results.dataset_loss_grads
        train_activations = prop_oppo_results.dataset_embed_activations
        target_labels = prop_oppo_results.dataset_target_labels

        num_samples = train_loss_grads.shape[0]

        # pre-compute proponents/opponents per training point
        train_idx__top_proponents = []
        train_idx__top_opponents = []

        n_jobs = psutil.cpu_count(logical=False)
        with ProgressParallel( n_jobs=n_jobs, prefer='threads' ) as parallel:
            def func_to_call( idx ):
                
                target_class = target_labels[ idx ]
                loss_grad = train_loss_grads[ idx ]
                activation = train_activations[ idx ]
                top_props, top_oppos = self.datapoint_output__top_prop_oppo(
                    target_class,
                    loss_grad,
                    activation,
                    prop_oppo_results,
                    top_k=top_k,
                ) # -> (5,4) (5,4)

                return top_props, top_oppos

            jobs = []
            for idx in range( num_samples ):
                jobs.append( joblib.delayed(func_to_call)(idx) )
                
            results = parallel( num_samples, jobs )

            train_idx__top_proponents = np.array( [ i[0] for i in results ] )
            train_idx__top_opponents  = np.array( [ i[1] for i in results ] )

        # print( 'output: ', train_idx__top_proponents.shape, train_idx__top_opponents.shape )

        return train_idx__top_proponents, train_idx__top_opponents
    
