import pytest
import os

from ddbg import DatasetDebugger
from ddbg.trainer import ModelTrainer, EmbeddingTrainer

@pytest.fixture(scope="module")
def mnist_project():
    ddbg_project = DatasetDebugger.from_recipe('mnist')
    return ddbg_project

def test_base_model_trainer( mnist_project ):

    cfg = mnist_project.cfg
    dataset = mnist_project.dataset
    model = mnist_project._init_model()

    model_trainer = ModelTrainer(
        cfg,
        dataset,
        model,
    )

    # basic checks of config init
    assert model_trainer.max_epochs == cfg.trainer.base.epochs, 'trainer config error!'

def test_embed_model_trainer( mnist_project ):

    cfg = mnist_project.cfg.clone()
    dataset = mnist_project.dataset
    model = mnist_project._init_model()

    # disable to avoid loading
    cfg.trainer.embedding.self_influence_percentile = 0
    cfg.trainer.embedding.pretrained_base_epoch = 0
    
    embed_trainer = EmbeddingTrainer(
        cfg,
        dataset,
        model,
        self_influence_results = None,
    )

    # basic checks of config init
    assert embed_trainer.max_epochs == cfg.trainer.embedding.epochs, 'embed trainer config error!'

    # check indices are loaded
    assert embed_trainer.self_influence_sorted_dataset_indices == None, 'self_influence_indices not none!'
    
