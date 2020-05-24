#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from sklearn.linear_model import LogisticRegression
from stability_selection import StabilitySelection
import numpy as np
from sklearn.pipeline import Pipeline

def execute_stability_selection(X, y):
    base_estimator = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    fs = StabilitySelection(base_estimator=base_estimator, verbose=1)
    fs.fit(X, y)
    scores = fs.stability_scores_.max(axis=1)

    # thresholding. threshold is score mean
    ranking = np.where(scores > np.mean(scores), scores, np.nan)
    rank_scores = np.unique(ranking[~np.isnan(ranking)])[::-1]
    
    for rank, rank_score in enumerate(rank_scores, start=1):
        ranking[ranking == rank_score] = rank

    return ranking

if __name__ == '__main__':
    run_pool(ranking_pool, execute_stability_selection, 'Stability Selection')
