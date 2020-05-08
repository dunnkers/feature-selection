from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import time
import os
from lib.feat_boost import FeatBoostClassification
from xgboost import XGBClassifier


def run_FeatBoost(enumerated):
    _, filename = enumerated
    # Load data
    test_data = pd.read_csv(filename, sep='\t')
    # Separate features/target
    features = test_data.drop('Class', axis=1).values
    labels = test_data['Class'].values

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
            siso_ranking_size = 8,\
            max_number_of_features = 100,\
            siso_order=4,\
            epsilon=1e-18,\
            verbose=2)
    print(filename)
    print('FeatBoost_XGBoost')

    # Run Feature Selection
    start = time.time()
    fs.fit(features, labels)
    elapsed = time.time() - start
    print('Completed scoring in {} seconds.'.format(elapsed))

    # feature names
    feature_names = test_data.drop('Class', axis=1).columns.values

    # Convert FeatBoost subset to ranking
    selected_features = np.unique(fs.selected_subset_) # FeatBoost duplicates bug
    ranking = list(map(lambda f: np.nan, feature_names)) # NaN values
    for rank, selected_feature in enumerate(selected_features, start=1):
        ranking[selected_feature] = rank

    # Write results to CSV
    output = pd.DataFrame(index=feature_names, data={\
        'FeatureRank': ranking,\
        'Time': elapsed\
    })
    return output