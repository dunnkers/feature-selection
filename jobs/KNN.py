from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from util import getDatasetNo, get_outpath
from sklearn.model_selection import cross_val_score

def runKNN(filename):
    clfname = 'KNN'
    dataset_dir = os.path.dirname(filename)

    # Load data
    test_data = pd.read_csv(filename, sep='\t')
    # Separate features/target
    features = test_data.drop('Class', axis=1).values
    labels = test_data['Class'].values


    # Load feature ranking
    inpath = '{}.csv'.format(get_outpath(dataset_dir, 'ranking'))
    results_data = pd.read_csv(inpath)
    rank_column = 'FeatureRank_{}'.format(getDatasetNo(filename))
    ranking_indices = results_data.sort_values(rank_column).index
    subsets = map(lambda i: (ranking_indices[i], ranking_indices[0:i+1]),\
        results_data.index)

    # Setup estimator
    neigh = KNeighborsClassifier(n_neighbors=1)
    
    # Run estimator for every subset (e.g. [1...n] features)
    acc_columns = 'Accuracy_{}_{}'.format(getDatasetNo(filename), clfname)
    for index, subset in subsets:
        X = np.take(features, subset, axis=1)
        accs = cross_val_score(neigh, X, labels)

        # store accuracy scores in dataframe; storing ALL folds
        for i, acc in enumerate(accs):
            acc_column = '{}_{}'.format(acc_columns, i + 1) # attach fold number
            results_data.loc[index, acc_column] = acc

    outpath = get_outpath(dataset_dir, 'validation')
    results_data.to_csv('{}.csv'.format(outpath), index=False)
