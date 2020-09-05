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

    ddbg_results = ddbg_project.calc_train_dataset__mislabel_scores()
    # ddbg_results = ddbg_project.load_ddbg_project_results()

    ddbg_viz = DdbgVisualize( ddbg_project )

    plts = ddbg_viz.visualize_top_self_influence( ddbg_results, end_idx=16 )

    plts = ddbg_viz.visualize_top_mislabel_score_items( ddbg_results, end_idx=16 )

    plt = ddbg_viz.visualize_dataset_embeddings( ddbg_results )



if __name__ == '__main__':
    main()
