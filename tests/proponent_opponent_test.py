import pytest
import os
import numpy as np

# from ddbg import DatasetDebugger
from ddbg.proponent_opponent import DatasetProponentOpponents, ProponentOpponentsResults


def test_dummy_proponent_opponent():

    prop_oppo_gen = DatasetProponentOpponents( [], None )

    assert len( prop_oppo_gen.models ) == 0, 'prop_oppo models error!'

def test_dummy_proponent_opponent_results():

    results = ProponentOpponentsResults(
        dataset_loss_grads = None,
        dataset_embed_activations = None,
        dataset_target_labels = np.arange( 10 ),
        final_predicted_labels = np.arange( 10 ),
        final_predicted_prob = np.arange( 10 ),
    )

    assert len( results.dataset_target_labels ) == 10
    assert len( results.final_predicted_labels ) == 10
    assert len( results.final_predicted_prob ) == 10
