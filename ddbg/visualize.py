import os, logging

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import umap

from ddbg import DatasetDebugger
from ddbg.ddbg_results import DdbgResults
from ddbg.self_influence import SelfInfluenceResults
from ddbg.proponent_opponent import ProponentOpponentsResults

ddbg_logger = logging.getLogger('ddbg')

class DdbgVisualize( object ):

    def __init__(
            self,
            ddbg_project : DatasetDebugger
    ):
        self.ddbg_project = ddbg_project
        self.cfg = ddbg_project.cfg
        self.dataset = ddbg_project.dataset

        
    def visualize_base_model_loss_curve( self, version='latest' ):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        log_dir = os.path.join( self.cfg.project.log_dir, '%s_base' % self.cfg.project.name )

        if version == 'latest':
            versions = [ int( item.split('_')[-1] ) for item in os.listdir(log_dir) ]
            versions.sort()
            ver_to_load = versions[-1]
        else:
            ver_to_load = version
            
        log_dir = os.path.join( log_dir, 'version_%s' % ver_to_load )
        
        ea = EventAccumulator( log_dir ).Reload()
        steps = []
        step_losses = []
        for event in ea.Scalars( 'train_loss' ):
            steps.append( event.step )
            step_losses.append( event.value )
            
        plt.figure(figsize=(10,10))
        plt.plot( steps, step_losses )
        plt.xlabel('Step')
        plt.ylabel('Loss')

        return plt
    

    def visualize_top_self_influence(
            self,
            ddbg_results: DdbgResults,            
            # self_influence_results: SelfInfluenceResults,
            # prop_oppo_results: ProponentOpponentsResults = None,
            start_idx: int = 0,
            end_idx: int = 8,
            rows: int = 8,
            sort_order: str = 'highest',
            color_map:  str = 'gray'
    ):
        dataset = self.dataset
        train_dataset, _ = dataset.get_datasets( self.cfg.dataset.dataset_path, None, None)

        # self_influence_scores = self_influence_results.self_influence_scores
        self_influence_scores = ddbg_results.self_influence_scores        
        indices = np.argsort(-self_influence_scores if sort_order=='highest' else self_influence_scores)

        plots = []
        for plot_offset in range (start_idx, end_idx, rows):
            indices_to_plot = indices[ plot_offset:plot_offset+rows ]
        
            plt = self.gen_item__prop_oppo_figure(
                indices_to_plot,
                train_dataset,
                ddbg_results,
                color_map = color_map,
            )
            plots.append( plt )

        return plots
        
    def gen_item__prop_oppo_figure(
            self,
            indices_to_plot,
            dataset,
            ddbg_results: DdbgResults,
            # self_influence_results: SelfInfluenceResults,
            # prop_oppo_results: ProponentOpponentsResults = None,
            color_map: str = 'gray',            
    ):
            
        def class_id__label( id ):
            return self.dataset.class__labels[ id ]
    
        # self_influence_scores = self_influence_results.self_influence_scores
        # predictions = self_influence_results.final_predicted_labels
        self_influence_scores = ddbg_results.self_influence_scores
        predictions = ddbg_results.final_predicted_classes

        # idx__top_proponents = prop_oppo_results.dataset_top_proponents if prop_oppo_results else None
        # idx__top_opponents  = prop_oppo_results.dataset_top_opponents if prop_oppo_results else None
        idx__top_proponents = ddbg_results.top_proponents
        idx__top_opponents  = ddbg_results.top_opponents
        
        self_influence_max = self_influence_scores.max()

        cols = 11
        rows = len( indices_to_plot )
        fig, subplots = plt.subplots(rows, cols, figsize=(12, 12))
        for i, data_index in enumerate( indices_to_plot ):
            data_point = dataset[ data_index ]
            image = data_point[0]
            target_label = class_id__label( data_point[1] )
            self_influence = self_influence_scores[ data_index ]
            prediction = predictions[ data_index ][0]
            predicted_label = class_id__label( prediction )
            
            row_idx=0
            subplot = subplots[i, row_idx]
            row_idx+=1
            
            imgplot = subplot.imshow(image, cmap=color_map)
            _=subplot.set_title( '%s(%s) %0.2f' % ( predicted_label, target_label, self_influence/self_influence_max) )
            _=subplot.set_xticks([])
            _=subplot.set_yticks([])

            if type( idx__top_proponents ) == type( None ): continue
            top_proponents = idx__top_proponents[ data_index ]
            top_opponents = idx__top_opponents[ data_index ]
            for item in list( top_proponents ) + list( top_opponents ):
                ddbg_logger.debug( 'col %d %s %0.3f %0.3f %0.3f ' % ( row_idx, item[0], item[1], item[2], item[3] ) ) # colum, idx, combined_scores, loss_grad_sim, activation_sim
                item_idx = int( item[0] )
                support_point = dataset[ item_idx ]
                support_image = support_point[0]
                support_label = class_id__label( support_point[1] )
                support_score = item[1]
                # support_predict_label = class_id__label( item[5] )
            
                subplot = subplots[i, row_idx]
                row_idx+=1
                imgplot = subplot.imshow(support_image, cmap=color_map)
                _=subplot.set_title( '(%s)' % ( support_label) )
                _=subplot.set_xticks([])
                _=subplot.set_yticks([])
            

        # plt.show()
        return plt

    def get_mislabel_indices_above_thresh(
            self,
            mislabel_scores: np.ndarray,
            mislabel_thresh:float = 0.5,            
    ) -> np.ndarray:
        
        sorted_indices = np.argsort( -mislabel_scores )

        above_idxes = np.where( mislabel_scores[ sorted_indices ] >= mislabel_thresh )[0]
        possible_mislabel_indices = sorted_indices[ above_idxes ]

        return possible_mislabel_indices


    def visualize_dataset_embeddings(
            self,
            ddbg_results: DdbgResults,            
            mislabel_thresh:float = 0.5,
            epoch_to_load: int = -1,
    ):

        possible_mislabel_indices = self.get_mislabel_indices_above_thresh( ddbg_results.mislabel_scores, mislabel_thresh )

        if epoch_to_load == -1:
            epoch_to_load = self.cfg.trainer.embedding.epochs-1
        embeddings, labels = self.ddbg_project._train_dataset_model__embeddings( epoch_to_load )
        
        if embeddings.shape[-1] > 2:
            embeddings_2d = self._project_2d_via_umap( embeddings )        
        else:
            embeddings_2d = embeddings

        plt = self.plot_embeddings( self.dataset.class__labels, embeddings_2d, labels )

        classes = len( self.dataset.class__labels )
        for class_id, label in enumerate( self.dataset.class__labels ):
            indices = np.where( labels[ possible_mislabel_indices ] == class_id )
            dataset_indices = possible_mislabel_indices[ indices ]
            plt.scatter(embeddings_2d[dataset_indices,0], embeddings_2d[dataset_indices,1], alpha=0.8, color='#000000', marker='$%s$' % label, zorder=10)

        # plt.show()
        return plt

    def visualize_top_mislabel_score_items(
            self,
            ddbg_results: DdbgResults,
            mislabel_thresh:float = 0.5,
            start_idx: int = 0,
            end_idx:   int = 32,
            rows:      int = 8,
    ):

        possible_mislabel_indices = self.get_mislabel_indices_above_thresh( ddbg_results.mislabel_scores, mislabel_thresh )
        
        train_dataset, _ = self.dataset.get_datasets( self.cfg.dataset.dataset_path, None, None)

        plots = []
        for plot_offset in range (start_idx, end_idx, rows):
            indices_to_plot = possible_mislabel_indices[ plot_offset:plot_offset+rows ]
            plt = self.gen_item__prop_oppo_figure(
                indices_to_plot,
                train_dataset,
                ddbg_results,
            )
            plots.append( plt )

        return plots
        
        
    def _project_2d_via_umap( self, embeddings, n_neighbors=10, min_dist=0.99, ):

        ddbg_logger.info( 'Projecting embedding via UMAP..' )
        umap_fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='euclidean',
        )
        embeddings_2d = umap_fit.fit_transform( embeddings )
        return embeddings_2d
        

    def plot_embeddings( self, class__labels, embeddings, targets, xlim=None, ylim=None):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

        plt.figure(figsize=(10,10))

        for i in range( len( class__labels ) ):
            inds = np.where(targets==i)[0]
            plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i], zorder=0)
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.legend( class__labels )
        
        return plt

