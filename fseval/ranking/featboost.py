#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
import numpy as np
from .lib.feat_boost import FeatBoostClassification
from xgboost import XGBClassifier

def FeatBoost_XGBoost(X, y):
    # Setup estimator
    xgboost_ensemble = XGBClassifier(max_depth=3, learning_rate=0.1,\
        n_estimators=200, silent=True, objective='binary:logistic',\
        booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,\
        max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,\
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,\
        random_state=0, seed=None, missing=None)
    # Setup FS method
    fs = FeatBoostClassification(estimator=[xgboost_ensemble,\
        xgboost_ensemble, xgboost_ensemble], number_of_folds = 10,\
            siso_ranking_size = 5,\
            max_number_of_features = 100,\
            siso_order=4,\
            epsilon=1e-18,\
            verbose=2)

    # Run Feature Selection
    fs.fit(X, y)

    # Convert FeatBoost subset to ranking
    selected_features = np.unique(fs.selected_subset_) # FeatBoost duplicates bug
    ranking = np.full(np.size(X, axis=1), np.nan) # empty array with NaN's
    for rank, selected_feature in enumerate(selected_features, start=1):
        ranking[selected_feature] = rank
    
    return ranking

if __name__ == '__main__':
    run_pool(ranking_pool, FeatBoost_XGBoost, 'FeatBoost_XGBoost')
