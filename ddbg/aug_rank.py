import os, logging, psutil
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms
import joblib

from ddbg import DatasetDebugger, DdbgVisualize
# from ddbg.models import model_name__model_class
# from ddbg.datasets import dataset_name__dataset_class

from ddbg.self_influence import DatasetSelfInfluence, SelfInfluenceResults
from ddbg.proponent_opponent import DatasetProponentOpponents, ProponentOpponentsResults, ProgressParallel


class AugRank( object ):

    def __init__(
            self,
            ddbg_project,
    ):
        self.ddbg_project = ddbg_project
        self.dataset = ddbg_project.dataset
        self.cfg = ddbg_project.cfg.clone()

        self.ddbg_results = ddbg_project.load_ddbg_project_results()
        self.baseline_self_influence_scores = self.ddbg_results.self_influence_scores
        
    def score_transform( self, transform_name, transform_to_score ):
        transforms_default = self.dataset.get_train_transform().transforms
        transforms_updated = [transform_to_score] + transforms_default
        transform = transforms.Compose( transforms_updated )
        
        transform_self_influence_results = self.calc_transform_self_influence( transform )

        file_name = os.path.join( '/tmp', 'xfm_%s.self_infl.pkl' % transform_name )
        transform_self_influence_results.save( file_name )

        # calc diff to baseline self_influence scores
        bss=self.baseline_self_influence_scores
        tss=transform_self_influence_results.self_influence_scores
        delta=(tss-bss)
        diff = np.log(np.abs(delta)+1)*delta/np.sqrt( bss*bss+tss*tss )
        
        print( '\nSI stats for %s' % transform_name, diff.sum(), diff.mean(), diff.max(), diff.min(), np.where(diff>0)[0].shape[0] )
        top_diff_indices = np.argsort( -diff )
        self.visualize_top_transform_self_influence_incr( transform_to_score, top_diff_indices[:8] )
        self.visualize_top_transform_self_influence_incr( transform_to_score, top_diff_indices[-8:] )

    def visualize_top_transform_self_influence_incr( self, transform_to_score, indices_to_plot ):
        
        ddbg_viz = DdbgVisualize( self.ddbg_project )
        train_dataset, _ = self.dataset.get_datasets( self.cfg.dataset.dataset_path, transform_to_score, None)
        
        ddbg_viz.gen_item__prop_oppo_figure(
                indices_to_plot,
                train_dataset,
                self.ddbg_results,
        )
        
    def calc_transform_self_influence(
            self,
            transform,
    ) -> SelfInfluenceResults:

        # self.logger.info( 'Calculating self influence..' )

        num_workers = psutil.cpu_count(logical=False)
        train_loader, _ = self.dataset.get_dataloaders(
            data_dir = self.cfg.dataset.dataset_path,
            batch_size = 64, # TODO, find safe batch size? self.cfg.trainer.base.batch_size,
            num_workers = num_workers,
            shuffle = False, # important to keep in data order
            train_transform = transform
        )

        self_influence_results = self.ddbg_project.calc_dataset_self_influence( train_loader, save_results=False )

        return self_influence_results
    
    def calc_transform_proponents(
            self,
            ddbg_project,
            transform_name,
            transform_to_score,
            top_k=5,
            exclude_indices=None,
    ):

        transforms_default = self.dataset.get_train_transform().transforms
        if transform_to_score:
            transforms_updated = [transform_to_score] + transforms_default
        else:
            transforms_updated = transforms_default
        transform = transforms.Compose( transforms_updated )

        orig_prop_oppo_results = ddbg_project.load_prop_oppo_results()
        
        num_workers = psutil.cpu_count(logical=False)
        train_loader, _ = self.dataset.get_dataloaders(
            data_dir = self.cfg.dataset.dataset_path,
            batch_size = 64, # TODO, find safe batch size? self.cfg.trainer.base.batch_size,
            num_workers = num_workers,
            shuffle = False, # important to keep in data order
            train_transform = transform
        )

        models = ddbg_project._load_base_model_checkpoints()        
        prop_oppo_gen = DatasetProponentOpponents(
            models,
            train_loader,
        )
        aug_prop_oppo_results = prop_oppo_gen.calc_proponent_opponent_results(
            precalc_all_prop_oppos=False,
        )
        
        # use outputs from orig trackin_gradients, calc proponents against augment  trackin_gradients
        orig_loss_grads = orig_prop_oppo_results.dataset_loss_grads
        orig_activations = orig_prop_oppo_results.dataset_embed_activations
        target_labels = orig_prop_oppo_results.dataset_target_labels

        num_samples = orig_loss_grads.shape[0]
        
        idx__top_proponents = []
        idx__top_opponents = []
        n_jobs = psutil.cpu_count(logical=False)

        with ProgressParallel( n_jobs=n_jobs, prefer='threads' ) as parallel:
            def func_to_call( idx ):
                target_class = target_labels[ idx ]
                loss_grad = orig_loss_grads[ idx ]
                activation = orig_activations[ idx ]
                
                top_props, top_oppos = prop_oppo_gen.datapoint_output__top_prop_oppo(
                    target_class,
                    loss_grad,
                    activation,
                    aug_prop_oppo_results,
                    top_k=top_k,
                    exclude_indices=exclude_indices,
                ) # -> (5,4) (5,4)
                return top_props, top_oppos
            
            jobs = []
            for idx in range( num_samples ):
                jobs.append( joblib.delayed(func_to_call)(idx) )
            
            results = parallel( len( jobs ), jobs )
            
            idx__top_proponents = np.array( [ i[0] for i in results ] )
            idx__top_opponents  = np.array( [ i[1] for i in results ] )

        aug_prop_oppo_results.set_top_prop_oppos(
                idx__top_proponents,
                idx__top_opponents,
        )

        file_name = os.path.join( '/tmp', 'xfm_%s.aug_prop_oppo.pkl' % transform_name )
        aug_prop_oppo_results.save( file_name )
        
        return aug_prop_oppo_results


    
'''
from ddbg.config.config import load_yaml_config
from ddbg import DatasetDebugger, DdbgVisualize
from ddbg.aug_rank import AugRank
import PIL

recipe='ddbg/recipes/mnist.yaml'
project_config = load_yaml_config( recipe )
ddbg_project = DatasetDebugger( project_config )

aug_rank = AugRank( ddbg_project )


import numpy as np
sorted_sis=np.argsort( aug_rank.baseline_self_influence_scores )
size_tenth = int(sorted_sis.shape[0]*.1)
sorted_sis_90p=sorted_sis[ -size_tenth: ]
sorted_sis_10p=sorted_sis[ :size_tenth ]

exclude_indices=sorted_sis_90p


#orig_prop_oppo_results = ddbg_project.load_prop_oppo_results()
#orig_idx__props=orig_prop_oppo_results.dataset_top_proponents

# load or compute orig props w/ exclude
transform_name = 'orig'
file_name = os.path.join( '/tmp', 'xfm_%s.aug_prop_oppo.pkl' % transform_name )
if os.path.exists( file_name ):
  from ddbg.proponent_opponent import ProponentOpponentsResults
  orig_prop_oppo_res = ProponentOpponentsResults.load( file_name )
else:
  orig_prop_oppo_res = aug_rank.calc_transform_proponents( ddbg_project, transform_name, None, exclude_indices=exclude_indices )
orig_idx__props=orig_prop_oppo_res.dataset_top_proponents

from torchvision import transforms

transform_name = 'randcrop_p4'
transform = transforms.RandomCrop(28,padding=4,fill=0,padding_mode='constant')
# 383814.9 6.396915 15.488129 -13.851119 54677

transform_name = 'randcrop_p1'
transform = transforms.RandomCrop(28,padding=1,fill=0,padding_mode='constant')
# 104283.33 1.7380555 14.5859585 -13.731639 43455


transform_name = 'rotate_d5'
transform = transforms.RandomAffine(degrees=5,fillcolor=0, resample=PIL.Image.BICUBIC)
# SI stats for rotate_d5 58610.91 0.9768485 14.384066 -13.733113 39478

transform_name = 'rotate_d15'
transform = transforms.RandomAffine(degrees=15,fillcolor=0, resample=PIL.Image.BICUBIC)
# SI stats for rotate_d15 85269.72 1.421162 14.588675 -13.7340975 41639

transform_name = 'shear_d15'
transform = transforms.RandomAffine(degrees=0, shear=15,fillcolor=0, resample=PIL.Image.BICUBIC)
# SI stats for shear_d15 67842.6 1.13071 15.115211 -14.067867 39998

transform_name = 'shear_d30'
transform = transforms.RandomAffine(degrees=0, shear=30,fillcolor=0, resample=PIL.Image.BICUBIC)
# SI stats for shear_d30 110932.125 1.8488687 14.687817 -13.733112 42964

import torch
import torchvision.transforms.functional as F
import random
from PIL import ImageEnhance

class TransformDarken:
    
    def __init__(self, brightness_range):
        self.brightness_range = brightness_range
    
    def __call__(self, img):
        brightness_factor = torch.tensor(1.0).uniform_(*self.brightness_range).item()
        # return F.adjust_brightness(img, brightness_factor)
        return  ImageEnhance.Brightness(img).enhance( brightness_factor)

class TransformBlur:
    def __init__(self, radius):
        self.radius = radius
    
    def __call__(self, img):
        return  img.filter(PIL.ImageFilter.GaussianBlur(self.radius))

class TransformRandomSqueeze:
    def __init__(self, max_pixels):
        self.max_pixels = max_pixels
    
    def __call__(self, img):
        pad_l = random.randint(1, self.max_pixels)
        pad_r = random.randint(1, self.max_pixels)
        pad_img = F.pad( img, (pad_l, 0, pad_r, 0) )
        return F.resize( pad_img, img.size[:2] )

class TransformRandomStretch:
    def __init__(self, max_pixels):
        self.max_pixels = max_pixels
    
    def __call__(self, img):
        crop_l = random.randint(1, self.max_pixels)
        crop_r = random.randint(1, self.max_pixels)
        img_size = img.size
        return F.resized_crop( img, 0, crop_l, img_size[1], img_size[0]-(crop_l+crop_r), img_size )


import cv2
class TransformErode:
    def __init__(self, size=3):
        self.kernel = np.ones((size,size),np.uint8)
    
    def __call__(self, img):
        img = cv2.erode(np.array(img),self.kernel,iterations = 1)
        return PIL.Image.fromarray( img )

class TransformDilate:
    def __init__(self, size=3):
        self.kernel = np.ones((size,size),np.uint8)
    
    def __call__(self, img):
        img = cv2.dilate(np.array(img),self.kernel,iterations = 1)
        return PIL.Image.fromarray( img )


# transform_name = 'darken_03.07'
# transform = transforms.ColorJitter(brightness=(0.03,0.07))


transform_name = 'blur_r2' # severe
transform = TransformBlur(radius=2)
# SI stats for blur_r2 372038.97 6.2006493 11.800753 -13.996237 56252

transform_name = 'blur_r1' # readable
transform = TransformBlur(radius=1)
# SI stats for blur_r1 230009.97 3.8334994 13.730929 -13.753653 53576
# aug_rank.score_transform( transform_name, transform )

transform_name = 'erode_s3'
transform = TransformErode(size=3)

transform_name = 'dilate_s3'
transform = TransformDilate(size=3)

transform_name = 'squeeze_s2'
transform = TransformRandomSqueeze(2)

transform_name = 'stretch_s2'
transform = TransformRandomStretch(2)


file_name = os.path.join( '/tmp', 'xfm_%s.aug_prop_oppo.pkl' % transform_name )
if os.path.exists( file_name ):
  from ddbg.proponent_opponent import ProponentOpponentsResults
  aug_prop_oppo_res = ProponentOpponentsResults.load( file_name )
else:
  aug_prop_oppo_res = aug_rank.calc_transform_proponents( ddbg_project, transform_name, transform, exclude_indices=exclude_indices )

aug_idx__props = aug_prop_oppo_res.dataset_top_proponents
mean_orig_idx__props = orig_idx__props.mean(axis=1)
mean_aug_idx__props = aug_idx__props.mean(axis=1)
diff=mean_aug_idx__props-mean_orig_idx__props # 1:score, 2: loss_sim, 3: act_sim

print( '\nProp stats for %s' % transform_name, '\n', diff[sorted_sis_90p,1:].mean(axis=0), '\n', diff[sorted_sis_10p,1:].mean(axis=0) )


Prop stats for randcrop_p4 
 [6.44949695e+03 3.15179862e-01 1.26812654e+05] 
 [3.52559933e-01 8.26433241e-07 2.89589518e+05]

Prop stats for randcrop_p1 
 [1.93632358e+03 8.99533881e-02 9.91238996e+03] 
 [1.92441420e-01 2.12140953e-07 9.03799594e+04]

Prop stats for shear_d30 
 [2.16917903e+03 1.54660323e-01 2.08975767e+04] 
 [1.62841129e-01 3.06868663e-07 1.05005531e+05]

Prop stats for stretch_s2 
 [6.16395054e+02 4.89900386e-02 -9.34179560e+03] 
 [2.29354397e-02 1.77531474e-07 1.06391560e+04]


Prop stats for squeeze_s2 
 [-2.78936356e+01  2.63252124e-03  6.05682997e+03] 
 [ 1.31189718e-03 -1.00417662e-07 -2.26319462e+04]

Prop stats for rotate_d15 
 [ 2.07313277e+03  1.23136989e-01 -1.05658381e+04] 
 [ 4.12570144e-02  5.37546109e-07 -6.61016957e+04]

Prop stats for dilate_s3 
 [ 8.52652778e+02  1.60450207e-01 -3.46848911e+04] 
 [-6.50812619e-02  2.39422215e-07 -1.33869933e+05]

Prop stats for erode_s3 
 [-5.61566310e+01  2.48856112e-01 -5.65269298e+04] 
 [-1.12362388e-01  1.93614482e-07 -2.37439701e+05]

Prop stats for blur_r1 
 [-1.77858909e+03  5.79276995e-02 -7.74225375e+04] 
 [-1.88992383e-01  3.61625870e-08 -2.93184434e+05]

Prop stats for blur_r2 
 [-2.72436529e+03  2.01463643e-01 -1.24932943e+05] 
 [-3.84945211e-01  8.94825713e-08 -4.89509526e+05]



import matplotlib.pyplot as plt
plt.show()

aug_rank = AugRank( ddbg_project )
self=aug_rank
transform_to_score=transform

aug_rank.visualize_top_transform_self_influence_incr(transform, [57057]*8); plt.show()
'''    
