import os
import argparse

from ddbg.config.config import load_yaml_config
from ddbg import DatasetDebugger, DdbgVisualize

def parse_args():

    parser = argparse.ArgumentParser('argument for ddbug demo')

    # yaml
    parser.add_argument('--recipe', type=str, default='ddbg/recipes/mnist.yaml')

    opt = parser.parse_args()
    return opt


def main():

    opt = parse_args()
    
    project_config = load_yaml_config( opt.recipe )
    ddbg_project = DatasetDebugger( project_config )

    ddbg_results = ddbg_project.calc_dataset__mislabel_scores()

    ddbg_viz = DdbgVisualize( ddbg_project )

    # ddbg_viz.visualize_base_model_loss_curve()

    self_influence_results = ddbg_project.load_self_influence_results()
    oppo_prop_results = ddbg_project.load_prop_oppo_results()
    ddbg_viz.visualize_top_self_influence( self_influence_results, oppo_prop_results )

    ddbg_viz.visualize_embeddings( self_influence_results, oppo_prop_results )



if __name__ == '__main__':
    main()
