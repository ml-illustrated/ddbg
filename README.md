# DatasetDebugger

Debugger for ML Datasets

## Features

```
from ddbg import DatasetDebugger
ddbg_project = DatasetDebugger.from_recipe('mnist')
ddbg_results = ddbg_project.calc_train_dataset__mislabel_scores()


from ddbg import DdbgVisualize
ddbg_viz = DdbgVisualize( ddbg_project )
# ddbg_results = ddbg_project.load_ddbg_project_results()

plts = ddbg_viz.visualize_top_self_influence( ddbg_results, end_idx=16 )

plts = ddbg_viz.visualize_top_mislabel_score_items( ddbg_results, end_idx=16 )

plt = ddbg_viz.visualize_dataset_embeddings( ddbg_results )

```

- TODO

## Credits

