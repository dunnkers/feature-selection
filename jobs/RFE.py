from sklearn.feature_selection import RFE
import pandas as pd
import time
import os
from xgboost import XGBClassifier

def runRFE(filename):
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
    fs = RFE(estimator=xgboost_ensemble, n_features_to_select=1, verbose=1)
    print(filename)
    print('RFE_XGBoost')

    # Run Feature Selection
    start = time.time()
    fs.fit(features, labels)
    elapsed = time.time() - start
    print('Completed scoring in {} seconds.'.format(elapsed))

    # Write results to CSV
    feature_names = test_data.drop('Class', axis=1).columns.values
    output = pd.DataFrame(index=feature_names, data={\
        'FeatureRank': fs.ranking_,\
        'Time': elapsed\
    })
    return output