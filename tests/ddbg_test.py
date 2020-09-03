import pytest
import os

from ddbg import DatasetDebugger

@pytest.fixture(scope="module")
def mnist_project():
    ddbg_project = DatasetDebugger.from_recipe('mnist')
    return ddbg_project

def test_load_mnist_project( mnist_project ):

    assert mnist_project != None, 'loading MNIST recipe failed!'

    config = mnist_project.cfg
    assert config.project.name == 'sample_mnist_embed2', 'yaml project name error!'
    assert config.dataset.name == 'mnist', 'yaml dataset name error!'
    assert config.model.arch == 'embed2',  'yaml model arch error!'

    assert os.path.exists( config.project.output_dir ), 'output dir missing!'


    from ddbg.datasets.mnist import MNISTWithIDs    
    dataset = mnist_project.dataset
    assert type( dataset ) == MNISTWithIDs, 'dataset incorrect class!'

def test_load_mnist_base_model( mnist_project ):
    model = mnist_project._init_model()

    assert model != None, 'base model loading error!'

    from ddbg.models.embed2 import Embed2Model
    assert type( model ) == Embed2Model, 'mnist base model is not an embed2 model!'

def test_load_mnist_embed_model( mnist_project ):
    embed_model = mnist_project._init_model( embed_mode=True )

    assert embed_model != None, 'embed model loading error!'

    from ddbg.models.embed2 import Embed2ModelEmbedOnly
    assert type( embed_model ) == Embed2ModelEmbedOnly, 'mnist embed model is not an embed2 model!'
    
