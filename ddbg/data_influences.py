import os, logging
import numpy as np

import torch
import torch.nn as nn

# https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian( y, x, create_graph=False ):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
        
    return torch.stack(jac).reshape(y.shape + x.shape)         


class DatasetSelfInfluence( object ):

    def __init__(
            self,
            self_influence_scores: np.ndarray,
            final_predicted_labels: np.ndarray,
            final_predicted_probs: np.ndarray,
    ):
        self.self_influence_scores = self_influence_scores
        self.final_predicted_labels = final_predicted_labels
        self.final_predicted_probs = final_predicted_probs

    def save( self, output_path ):
        out_dir = os.path.split( output_path )[0]
        os.makedirs( out_dir, exist_ok=True )
            
        with open( output_path, 'wb' ) as fp:
            np.save( fp, self.self_influence_scores )
            np.save( fp, self.final_predicted_labels )
            np.save( fp, self.final_predicted_probs )

    @classmethod
    def load( klass, load_path ):
            
        with open( load_path, 'rb' ) as fp:
            self_influence_scores = np.load( fp )
            final_predicted_labels = np.load( fp )
            final_predicted_probs = np.load( fp )

        return klass(
            self_influence_scores,
            final_predicted_labels,
            final_predicted_probs,
        )

            

class CalcSelfInfluence( object ):

    def __init__( self ):
        self.models = models
        self.func_model__monitor_layer = func_model__monitor_layer # ?
        self.data_loader = data_loader

        self.logger = logging.getLogger('')

    def run( self ) -> DatasetSelfInfluence:

        all_influence_scores = []
        all_final_output_probs = []
        all_final_predicted_labels = []
        for idx, (inputs, targets, input_indexes) in enumerate( self.data_loader ):
            if (idx % 50) == 0:
                self.logger.info( 'self_influence batch %s' % idx )
            # inputs.shape:  torch.Size([64, 3, 32, 32])
            # targets.shape: torch.Size([64])
            
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
        
            self_influence_outputs = self._input_batch__self_influence_scores( inputs, targets )
            self_influence_scores, final_output_probs, final_predicted_labels = self_influence_outputs
        
            all_influence_scores.append( self_influence_scores.detach().cpu().numpy().copy() )
            all_final_output_probs.append( final_output_probs.detach().cpu().numpy().copy() )
            all_final_predicted_labels.append( final_predicted_labels.detach().cpu().numpy().copy() )
            del inputs
            del targets
            del self_influence_scores
            del final_output_probs
            del final_predicted_labels
        
        self_influence_scores = np.concatenate( all_influence_scores )
        final_output_probs = np.concatenate( all_final_output_probs )
        final_predicted_labels = np.concatenate( all_final_predicted_labels )

        dataset_self_influence = DatasetSelfInfluence( 
            self_influence_scores = self_influence_scores,
            final_predicted_labels = final_predicted_labels,
            final_predicted_probs = final_output_probs,
        )
        return dataset_self_influence
            
    def _per_model_self_influence_scores(
            self,
            model,
            monitor_params,
            inputs,
            targets
    ):
    
        cls_outputs = model(inputs) # -> [64, 100]
        # compute loss
        loss = model.criterion_no_reduction(cls_outputs, targets ) # -> (batch,)

        # calc d_layer / d_loss
        scores = []
        for monitor_param in monitor_params:
            jacobians = jacobian(loss, monitor_param) # (batch, *param.shape)
            score = torch.sum( jacobians*jacobians, list(range(1, jacobians.ndim)) ) # sum across non batch dims -> (batch,)
            scores.append( score )
            
        scores = torch.sum( torch.stack( scores, axis=-1 ), axis=-1 ) # (batch, 2) -> (batch,)
        
        return scores, cls_outputs
    

    def _input_batch__self_influence_scores(
            self,
            inputs,
            targets,
    ):
    
        self_influence_scores = []
        for model in self.models:
            monitor_params = self.func_model__monitor_layer( model )
            scores, cls_outputs = self._per_model_self_influence_scores(
                model = model,
                monitor_params = monitor_params,
                inputs = inputs,
                targets = targets
            )
            self_influence_scores.append( scores )

        # use final model's outputs
        final_output_probs, final_predicted_labels = torch.topk( cls_outputs, 1 )

        across_models_self_influence_scores = torch.sum( torch.stack( self_influence_scores, axis=-1 ), axis=-1 )
        return across_models_self_influence_scores, final_output_probs, final_predicted_labels

    
