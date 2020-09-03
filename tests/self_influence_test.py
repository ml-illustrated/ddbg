import pytest
import os
import numpy as np

# from ddbg import DatasetDebugger
from ddbg.self_influence import DatasetSelfInfluence, SelfInfluenceResults


def test_dummy_self_influence():

    def dummy_func():
        return []
    
    self_infl_gen = DatasetSelfInfluence( [], dummy_func, None )

    assert len( self_infl_gen.models ) == 0, 'self_influence models error!'

def test_dummy_self_influence_results():

    results = SelfInfluenceResults(
        self_influence_scores = np.arange( 10 ),
        final_predicted_labels = np.arange( 10 ),
        final_predicted_probs = np.arange( 10 )
    )

    assert len( results.self_influence_scores ) == 10
    assert len( results.final_predicted_labels ) == 10
    assert len( results.final_predicted_probs ) == 10
