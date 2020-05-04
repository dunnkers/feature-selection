from FeatBoost import FeatBoostClassification
import pandas as pd
import sys
import os
import random
import math
import numpy as np
from warnings import warn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import scipy.io as sio
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from scipy import stats
from xgboost import XGBClassifier

def shuffle_in_unison(a,b):
    p = np.random.permutation(len(a))
    return a[p],b[p]

Experiment_name = "Spambase"
file_name = "datasets/Spam_Input"
dataset = pd.read_csv(file_name, header=None)
dataset_array = dataset.values
train_input_test = dataset_array[:,:len(dataset_array[0])-1]
train_output_d_test = dataset_array[:,len(dataset_array[0])-1:]

samples = len(train_input_test)
number_of_features = len(train_input_test[0])
informative_features = "NA"
redundant_features = "NA"
repeated_features = "NA"
class_separation = "NA"
number_of_classes = 2

"""
Experiment_name = "Mfeat Dataset"
train_input_test = sio.loadmat('datasets/mfeat/input_mfeat.mat')
train_output_d_test= sio.loadmat('datasets/mfeat/output_mfeat.mat')
train_input_test = train_input_test['input']
train_output_d_test = train_output_d_test['output']

samples = len(train_input_test)
number_of_features = len(train_input_test[0])
informative_features = "NA"
redundant_features = "NA"
repeated_features = "NA"
class_separation = "NA"
number_of_classes = 2
"""

"""
Experiment_name = "BASEHOCK"
input_test = sio.loadmat('datasets/BASEHOCK.mat')
train_input_test = input_test['X']
train_output_d_test = input_test['Y']

samples = len(train_input_test)
number_of_features = len(train_input_test[0])
informative_features = "NA"
redundant_features = "NA"
repeated_features = "NA"
class_separation = "NA"
number_of_classes = 2
"""

train_input_test = stats.zscore(train_input_test)

stopping_criteria = []

train_input_test,train_output_d_test = shuffle_in_unison(train_input_test,train_output_d_test)

number_of_training_samples = int(samples * 0.70)
test_input = train_input_test[number_of_training_samples:,:]
test_output_d = train_output_d_test[number_of_training_samples:]
number_of_test_samples = len(test_input)

if(number_of_training_samples > 4000):
    limit_train_samples = 4000
else:
    limit_train_samples = number_of_training_samples

train_input = train_input_test[:limit_train_samples,:]
train_output_d = train_output_d_test[:limit_train_samples]

# IIS-C
# initialize array for saving results of multiple runs
if(number_of_features > 10):
    max_features = 10
else:
    max_features = number_of_features

iterations = 10
ranking_size = 8
residual_size = iterations*max_features

#IIS
all_accuracies = 99*np.ones((iterations,max_features))
residuals = np.zeros((residual_size,len(train_output_d)))
siso_ranking = 99*np.ones((residual_size,ranking_size))
selected_feature = 99*np.ones((iterations,max_features))
all_subsets = 99*np.ones((iterations,max_features))
iis_accuracies = 99*np.ones((iterations,max_features))
iis_accuracies_xgb = 99*np.ones((iterations,max_features))
all_stopping = []
all_random_accuracies = []

xgboost_ensemble = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=200, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
ET_ensemble = ExtraTreesClassifier(n_estimators=200,min_samples_split=5, max_features= 'auto', oob_score=True, bootstrap=True)
RF_ensemble = RandomForestClassifier(n_estimators=200,min_samples_split=5, max_features= 'auto', oob_score=True, bootstrap=True)

for i in range(0,iterations):

    n_inputs = train_input.shape[1]

    #Iterative Input Selection Subset
    a = FeatBoostClassification(estimator = [xgboost_ensemble, xgboost_ensemble, xgboost_ensemble], number_of_folds = 10, siso_ranking_size = ranking_size, max_number_of_features = max_features, siso_order=4, epsilon=1e-18, verbose=2)
    x = a.fit(X = train_input, Y = np.ravel(train_output_d))

    n = len(a.accuracy_)
    all_accuracies[i,0:n] = a.accuracy_
    all_subsets[i,0:n] = a.selected_subset_

    print "iteration %02d finished" % (i+1)


kf = KFold(n_splits=10, shuffle = False)
KNN = KNeighborsClassifier(n_neighbors = 5)

#Iterative Input Selection Accuracy
for i in range(0,iterations):
    features = []
    for j in range(0,len(all_subsets[i])):
        if(all_subsets[i][j] != 99.):
            features.append(int(all_subsets[i][j]))
            iis_accuracies[i][j] = np.mean(cross_val_score(KNN, test_input[:,features], np.ravel(test_output_d), cv=kf))
            iis_accuracies_xgb[i][j] = np.mean(cross_val_score(xgboost_ensemble, test_input[:,features], np.ravel(test_output_d), cv=kf))
print "IIS Accuracy Calculated"
print iis_accuracies
print "IIS Accuracy with XGB Calculated"
print iis_accuracies_xgb
