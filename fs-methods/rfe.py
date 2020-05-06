from sklearn.feature_selection import RFE
import pandas as pd
import sys
import time
from xgboost import XGBClassifier

# Load data
filename = sys.argv[1]
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

fs = RFE(estimator=xgboost_ensemble, n_features_to_select=1)

print(filename)
print('RFE_XGBoost')

start = time.time()

fs.fit(features, labels)

elapsed = time.time() - start
print('Completed scoring in {} seconds.'.format(elapsed))


feature_importances = pd.DataFrame(data={'FeatureName': test_data.drop(\
                                            'Class', axis=1).columns.values,
                                         'FeatureImportance': fs.ranking_})

for i, row in feature_importances.sort_values('FeatureImportance', ascending=True).iterrows():
    print('{}\t{}'.format(row['FeatureName'], row['FeatureImportance']))
