import os
import argparse

from ddbg.config.config import load_yaml_config
# from ddbg import DatasetDebugger, DdbgVisualize
from ddbg.active_learning import ActiveLearner

def parse_args():

    parser = argparse.ArgumentParser('argument for ddbg demo')

    # yaml
    parser.add_argument('--recipe', type=str, default='ddbg/recipes/active_learning/mnist_random.yaml')

    opt = parser.parse_args()
    return opt


def main():

    opt = parse_args()
    
    project_config = load_yaml_config( opt.recipe )
    ddbg_active_learner = ActiveLearner( project_config )

    ddbg_active_learner.run_active_learning_simulation()


if __name__ == '__main__':
    main()
