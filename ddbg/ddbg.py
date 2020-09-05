import os, logging, psutil
from typing import Tuple

import numpy as np
import torch

from ddbg.models import model_name__model_class
from ddbg.datasets import dataset_name__dataset_class

from ddbg.trainer import ModelTrainer, EmbeddingTrainer
from ddbg.self_influence import DatasetSelfInfluence, SelfInfluenceResults
from ddbg.proponent_opponent import DatasetProponentOpponents, ProponentOpponentsResults

from ddbg.ddbg_results import DdbgResults

class DatasetDebugger( object ):

    def __init__(
            self,
            project_config,
            dataset = None,
            model_name = None,
    ):
        self.cfg = project_config
        self.dataset = dataset if dataset else None

        self.model_name = self.model_name.lower() if model_name else self.cfg.model.arch.lower()

        self._init_dataset()

        # print( self.model )
        # print( 'project_config:\n%s' % self.cfg )

        os.makedirs( self.cfg.project.output_dir, exist_ok=True )
        output_path = os.path.join( self.cfg.project.output_dir, self.cfg.file_names.save_config )
        with open( output_path, 'w' ) as fp:
            fp.write( '%s' % self.cfg )
        
        self._setup_logging()

    @classmethod
    def from_recipe( klass, recipe_name: str ):
        import inspect
        from ddbg.config.config import load_yaml_config

        if not recipe_name.endswith( '.yaml' ):
            recipe_name = '%s.yaml' % recipe_name
        recipe_path = os.path.join( os.path.dirname(inspect.getfile( DatasetDebugger )), 'recipes', recipe_name )
        project_config = load_yaml_config( recipe_path )
        ddbg_project = klass( project_config )

        return ddbg_project
        
    def calc_train_dataset__mislabel_scores(
            self,
            precalc_all_prop_oppos:bool = True,
    ) -> DdbgResults:
        '''
        Primary function to compute a dataset's mislabel stats
        '''

        self.logger.info( 'Start dataset analysis..' )
        self.train_base_model()
        self.logger.info( 'Training base model done.')

        self_influence_results = self.calc_dataset_self_influence()
        prop_oppo_results = self.calc_dataset_proponent_opponents( precalc_all_prop_oppos=precalc_all_prop_oppos )
        '''
        self_influence_results = self.load_self_influence_results()
        prop_oppo_results = self.load_prop_oppo_results()
        '''

        self.train_embedding_model( self_influence_results )
        self.logger.info( 'Training embedding model done.')

        mislabel_scores, centroid_embeddings = self.calc_train_dataset_mislabel_scores( self_influence_results )

        ddbg_results = DdbgResults(
            centroid_embeddings = centroid_embeddings,
            mislabel_scores = mislabel_scores,
            self_influence_scores = self_influence_results.self_influence_scores,
            final_predicted_classes = self_influence_results.final_predicted_labels, # TODO: recompute?
            top_proponents = prop_oppo_results.dataset_top_proponents,
            top_opponents = prop_oppo_results.dataset_top_opponents,
        )

        output_path = os.path.join( self.cfg.project.output_dir, self.cfg.file_names.project_results )
        ddbg_results.save( output_path )

        self.logger.info( 'Dataset analysis complete and saved to %s.' % output_path )        
        
        return ddbg_results

    def load_ddbg_project_results( self ):
        load_path = os.path.join( self.cfg.project.output_dir, self.cfg.file_names.project_results )
        ddbg_results = DdbgResults.load( load_path )
        
        return ddbg_results
        
        
    def train_base_model( self ):
        model = self._init_model()

        model_trainer = ModelTrainer(
            self.cfg,
            self.dataset,
            model,
        )
        model_trainer.train()
        model_trainer.eval_base_model_metric()

    def calc_dataset_self_influence(
            self,
            use_train_dataset: bool = True,
    ) -> SelfInfluenceResults:

        self.logger.info( 'Calculating self influence..' )
        
        models = self._load_base_model_checkpoints()

        train_loader, test_loader = self._get_dataset_for_data_influence()
        data_loader = train_loader if use_train_dataset else test_loader

        embed_layer_name = self.cfg.model.embed_layer_name

        def func_model__monitor_layer( model ):
            return [ param for name, param in model.named_parameters() if name.startswith( embed_layer_name ) ]
        
        self_infl_gen = DatasetSelfInfluence( 
            models,
            func_model__monitor_layer,
            data_loader = data_loader,
        )
        
        self_influence_results = self_infl_gen.calc_self_influence_results()

        if self.cfg.data_influence.self_influence_path:
            file_name = self.cfg.file_names.train_dataset_self_influence_results if use_train_dataset else self.cfg.file_names.test_dataset_self_influence_results
            file_name = os.path.join( self.cfg.data_influence.self_influence_path, file_name )
            self_influence_results.save( file_name )

        return self_influence_results

    def calc_dataset_proponent_opponents(
            self,
            use_train_dataset: bool = True,            
            precalc_all_prop_oppos:bool = True,
    ) -> ProponentOpponentsResults:

        self.logger.info( 'Calculating proponents & opponents..' )
        
        models = self._load_base_model_checkpoints()

        train_loader, test_loader = self._get_dataset_for_data_influence()
        data_loader = train_loader if use_train_dataset else test_loader
        
        prop_oppo_gen = DatasetProponentOpponents(
            models,
            data_loader,
        )

        prop_oppo_results = prop_oppo_gen.calc_proponent_opponent_results(
            precalc_all_prop_oppos=precalc_all_prop_oppos,
        )

        if self.cfg.data_influence.prop_oppos_path:
            file_name = self.cfg.file_names.train_dataset_prop_oppos_results if use_train_dataset else self.cfg.file_names.test_dataset_prop_oppos_results
            file_name = os.path.join( self.cfg.data_influence.prop_oppos_path, file_name )
            prop_oppo_results.save( file_name )

        return prop_oppo_results

    def train_embedding_model(
            self,
            self_influence_results: SelfInfluenceResults = None,
    ):
        self.logger.info( 'Training embed model..' )
        
        embed_model = self._init_model( embed_mode=True )
        embedding_trainer = EmbeddingTrainer(
            self.cfg,
            self.dataset,
            embed_model,
            self_influence_results,
        )
        embedding_trainer.train()

    def calc_train_dataset_mislabel_scores(
            self,
            self_influence_results: SelfInfluenceResults,            
    ) -> Tuple[ np.ndarray, np.ndarray ]:

        self.logger.info( 'Generating embeddings..' )

        last_embed_epoch = self.cfg.trainer.embedding.epochs-1
        embeddings, labels = self._train_dataset_model__embeddings( last_embed_epoch )

        self.logger.info( 'Calculating mislabel scores..' )        
        num_classes = self.dataset.num_classes
        class_id__centroid_embedding = self._calc_centroid_embeddings( num_classes, embeddings, labels )

        embed_idx__dist_from_centroid = self._calc_embed__centroid_dist( class_id__centroid_embedding, embeddings, labels )

        idx__mislabel_score = self._calc_mislabel_scores( self_influence_results, embed_idx__dist_from_centroid )

        '''
        if self.cfg.project.output_dir:
            file_name = os.path.join( self.cfg.project.output_dir, self.cfg.file_names.train_dataset_mislabel_scores )
            with open( file_name, 'wb' ) as fp:
                np.save( fp, idx__mislabel_score )
        '''
        
        return idx__mislabel_score, class_id__centroid_embedding

    def _calc_mislabel_scores(
            self,
            self_influence_results: SelfInfluenceResults,
            embed_idx__dist_from_centroid: np.ndarray,
    ) -> np.ndarray:
            
        self_influence_scores = self_influence_results.self_influence_scores
        self_influence_scores_normed = self_influence_scores / self_influence_scores.max()
        # idx__mislabel_score = embed_idx__dist_from_centroid*self_influence_scores_normed
        # idx__mislabel_score = self_influence_scores_normed # top rank mislabels, lower appers to be confusions by opponents
        # idx__mislabel_score = embed_idx__dist_from_centroid # mixture of unusuals and mislabels
        idx__mislabel_score = embed_idx__dist_from_centroid+self_influence_scores_normed # top rank mislabels, then dominated by unusuals

        return idx__mislabel_score

    def _calc_centroid_embeddings(
            self,
            num_classes: int,
            embeddings:  np.ndarray,
            target_labels:np.ndarray,
    ) -> np.ndarray:
                                  
        class_id__centroid_embedding = []
        for class_id in range( num_classes ):
            indicies = np.where( target_labels == class_id )[0]
            class_embeddings = embeddings[ indicies ]
            centroid_embedding = class_embeddings.mean( axis=0 )        
            centered = class_embeddings - centroid_embedding
            dist_to_centroid = np.linalg.norm( centered, axis=1 )
            # do quick outlier rejection
            within_cluster = class_embeddings[ dist_to_centroid < dist_to_centroid.mean() + 2*dist_to_centroid.std() ]
            self.logger.debug( 'filtered %s centroid from %s to %s' % ( class_id, len( class_embeddings ), len( within_cluster ) ) )
            centroid_embedding = within_cluster.mean( axis=0 )                
            class_id__centroid_embedding.append( centroid_embedding )

        return np.array( class_id__centroid_embedding )
    
    def _calc_embed__centroid_dist(
            self,
            class_id__centroid_embedding: np.ndarray,
            embeddings:  np.ndarray,
            target_labels:np.ndarray,
    ) -> np.ndarray:
                                  
        embed_idx__dist_from_centroid = np.zeros( embeddings.shape[0], dtype=np.float )    
        for class_id, centroid_embedding in enumerate( class_id__centroid_embedding ):
            indicies = np.where( target_labels == class_id )[0]
            class_embeddings = embeddings[ indicies ]

            # calc max dist to updated centroid
            centered = class_embeddings - centroid_embedding
            dist_to_centroid = np.linalg.norm( centered, axis=1 )
            max_dist_to_centroid = dist_to_centroid.max()
            dist_threshold = dist_to_centroid.mean() + 2*dist_to_centroid.std()
            norm_dist = max_dist_to_centroid - dist_threshold
            outside_indicies = np.where( dist_to_centroid >= dist_threshold )[0]
            dataset_indicies = indicies[ outside_indicies ]
            embed_idx__dist_from_centroid[ dataset_indicies ] = ( dist_to_centroid[ outside_indicies ] - dist_threshold ) / norm_dist

        return embed_idx__dist_from_centroid
    
    def _load_base_model_checkpoints( self ) -> list:

        epochs_to_load = self.cfg.data_influence.checkpoint_epochs

        model_name = self.cfg.model.arch.lower()
        
        Model_Class = model_name__model_class[ model_name ]
        raw_models = Model_Class.load_training_checkpoints(
            self.cfg,
            epochs_to_load,
            input_channels = self.dataset.input_channels,
            input_size = self.dataset.input_size
        )
        
        models = []
        for model in raw_models:
            models.append( model.to('cuda') if torch.cuda.is_available() else model )

        return models
        
    def _setup_logging( self ):
        log_file_name = os.path.join( self.cfg.project.output_dir, 'train_%s.log' % self.cfg.project.name )
        log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        formatter = logging.Formatter( log_format )
        
        self.logger = logging.getLogger('ddbg')
        self.logger.setLevel(logging.INFO)
        
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        file_handler = logging.FileHandler( log_file_name )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # save lightning logs to file also
        lightning_logger = logging.getLogger('lightning')
        lightning_logger.addHandler(file_handler)

        # self.logger.info( 'project_config:\n%s' % self.cfg )
        
    def _init_dataset( self ):
        if self.dataset != None: return

        dataset_name = self.cfg.dataset.name.lower()
        if dataset_name == 'custom':
            raise ValueError( 'dataset must be passed in for type "custom"' )
        elif dataset_name not in dataset_name__dataset_class:
            raise NotImplementedError( 'Unknown dataset name: choices are %s' % '|'.join( dataset_name__dataset_class.keys() ) )
        
        Dataset_class = dataset_name__dataset_class[ dataset_name ]
        self.dataset = Dataset_class()

    def _init_model( self, embed_mode=False ):

        model_name = self.model_name

        if model_name == 'custom':
            # raise ValueError( 'model must be passed in for type "custom"' )
            raise NotImplementedError( 'custom model not supported yet' )
        elif model_name not in model_name__model_class:
            raise NotImplementedError( 'Unknown model arch: choices are %s' % '|'.join( model_name__model_class.keys() ) )

        if embed_mode:
            model_name += 'embed_only' # hack
        
        Model_Class = model_name__model_class[ model_name ]
        model = Model_Class.get_default_model(
            self.cfg,
            num_classes = self.dataset.num_classes,
            input_size = self.dataset.input_size,
            input_channels = self.dataset.input_channels
        )
        return model
        
    def _get_dataset_for_data_influence( self ):
        num_workers = psutil.cpu_count(logical=False)
        train_loader, test_loader = self.dataset.get_dataloaders(
            data_dir = self.cfg.dataset.dataset_path,
            batch_size = 64, # TODO, find safe batch size? self.cfg.trainer.base.batch_size,
            num_workers = num_workers,
            shuffle = False, # important to keep in data order
        )
        return train_loader, test_loader

    def _train_dataset_model__embeddings(
            self,
            embed_epoch: int
    ) -> Tuple[ np.ndarray, np.ndarray ]:

        embed_model = self._init_model( embed_mode=True )
        # embed_model.load_pretrained_base_model( 11 ) # visualize base model
        embed_model.load_pretrained( epoch=embed_epoch )

        train_loader, _ = self._get_dataset_for_data_influence()

        embeddings, labels = self._extract_embeddings(train_loader, embed_model)
        return embeddings, labels

    def _extract_embeddings(
            self,
            dataloader,
            embed_model
    ) -> Tuple[ np.ndarray, np.ndarray ]:

        if torch.cuda.is_available():
            embed_model = embed_model.cuda()
        with torch.no_grad():
            embed_model.eval()
            embeddings = []
            labels = np.zeros(len(dataloader.dataset), dtype=np.int)
            k = 0
            for images, targets, img_idxes in dataloader:
                if torch.cuda.is_available():
                    images = images.cuda()
                embeddings.append( embed_model(images).data.cpu().numpy() )
                labels[k:k+len(images)] = targets.numpy()
                k += len(images)
                
        embeddings = np.concatenate( embeddings )
        # print( 'extract_embed: ', labels[debug_idx], dataloader.dataset[debug_idx][1:] )
           
        return embeddings, labels

    def load_self_influence_results(
            self,
            load_train_data:bool = True,
    ) -> SelfInfluenceResults:

        file_name = self.cfg.file_names.train_dataset_self_influence_results if load_train_data else self.cfg.file_names.test_dataset_self_influence_results        
        file_path = os.path.join( self.cfg.data_influence.self_influence_path, file_name )
        self_influence_results = SelfInfluenceResults.load( file_path )
        
        return self_influence_results

    def load_prop_oppo_results(
            self,
            load_train_data:bool = True,
    ) -> ProponentOpponentsResults:

        file_name = self.cfg.file_names.train_dataset_prop_oppos_results if load_train_data else self.cfg.file_names.test_dataset_prop_oppos_results
        file_path = os.path.join( self.cfg.data_influence.prop_oppos_path, file_name )
        prop_oppo_results = ProponentOpponentsResults.load( file_path )
        
        return prop_oppo_results

    '''
    def load_train_dataset_mislabel_score( self ):

        file_name = os.path.join( self.cfg.project.output_dir, self.cfg.file_names.train_dataset_mislabel_scores )
        with open( file_name, 'rb' ) as fp:
            idx__mislabel_score = np.load( fp )

        return idx__mislabel_score
    '''
