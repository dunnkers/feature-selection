#!/usr/bin/env python
from ComputePool import run_ranking_pool
from skrebate import MultiSURF as MultiSURFAlgorithm
import numpy as np

def MultiSURF(X, y):
    # Setup FS method
    dataset_features = np.size(X, axis=1) # rank all features
    fs = MultiSURFAlgorithm(n_features_to_select=dataset_features)

    # Run Feature Selection
    fs.fit(X, y)

    return fs.top_features_

if __name__ == '__main__':
    run_ranking_pool(MultiSURF)
