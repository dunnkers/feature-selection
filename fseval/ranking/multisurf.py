#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from skrebate import MultiSURF
import numpy as np

def execute_MultiSURF(X, y):
    # Setup FS method
    dataset_features = np.size(X, axis=1) # rank all features
    fs = MultiSURF(n_features_to_select=dataset_features, verbose=True)

    # Run Feature Selection
    fs.fit(X, y)

    return fs.top_features_

if __name__ == '__main__':
    run_pool(ranking_pool, execute_MultiSURF, 'MultiSURF')
