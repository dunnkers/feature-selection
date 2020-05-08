from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from util import get_outpath
from sklearn.model_selection import cross_val_score

def run_validation(enumerated):
    key, filename = enumerated
    dataset_dir = os.path.dirname(filename)
    print('Validating set #{} from {}'.format(key, dataset_dir))

    # Load data
    test_data = pd.read_csv(filename, sep='\t')

    # Load feature ranking
    inpath = '{}.csv'.format(get_outpath(dataset_dir, 'ranking'))
    results_data = pd.read_csv(inpath, header=[0,1], index_col=0)
    current_column = results_data[str(key)].copy()
    ranking = current_column.sort_values('FeatureRank').index
    # -> e.g. ['feature_name', 'another_feature', ...] in order of feature rank

    # Construct subsets
    subsets = map(lambda i: (ranking[i], ranking[0:i+1]), range(len(ranking)))
    # -> e.g. [[<best ranked feature>], [<top 2 ranked features>], ... etc]

    # Setup validation estimators
    clfs = [('KNN', KNeighborsClassifier(n_neighbors=1))]
    
    # Run all estimators
    for clfname, estimator  in clfs:
        # Run estimator for every subset
        for index, subset in subsets:
            X = test_data[subset].values # pandas multi-column select
            y = test_data['Class'].values
            scores = cross_val_score(estimator, X, y)

            # store accuracy scores in dataframe; storing ALL folds
            for i, acc in enumerate(scores):
                acc_column = 'Accuracy_{}_{}'.format(clfname, i + 1) # fold number
                current_column.loc[index, acc_column] = acc

    print('Set #{} validated using {} estimator(s).'.format(key, len(clfs)))
    return current_column
