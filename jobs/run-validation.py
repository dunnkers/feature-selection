#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import pandas as pd
from util import get_datasets, get_poolSize, get_outpath



from sklearn.neighbors import KNeighborsClassifier
import os

def runKNN(filename):
    # Load data
    test_data = pd.read_csv(filename, sep='\t')
    features = test_data.drop('Class', axis=1).values
    labels = test_data['Class'].values

    # Load results data
    outpath = get_outpath(os.path.dirname, 'results_ranking')
    results_data = pd.read_csv(outpath)

    # Setup estimator
    neigh = KNeighborsClassifier(n_neighbors=1)

    # Run validation
    start = time.time()
    # accs = cross_val_score(neigh, X_new, y) # 5-fold cv
    elapsed = time.time() - start
    print('Completed validation in {} seconds.'.format(elapsed))

    # Write results to CSV
    # results_data['Accuracy_KNN'] = accs
    results_data.to_csv('results/{}'.format(os.path.basename(filename)),\
        index=False)
    return results_data

if __name__ == '__main__':
    begin = time.time()
    # split up in data_dir -> results_dir?
    dataset_dir = sys.argv[1]

    inputs = get_datasets(dataset_dir)
    poolSize = get_poolSize()

    # start pool
    pool = multiprocessing.Pool(processes=poolSize,)
    poolResults = pool.map(lambda input: print(input), inputs) # Do the calculation.

    # collect results
    pool.close() # Stop pool accordingly.
    pool.join()  # Wrap up data from the workers in the pool.
    results = pd.concat(poolResults, axis=1)
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , poolSize, ' workers: ', elapsedTime, ' seconds')

    # write to csv
    outpath = get_outpath(dataset_dir, 'results_validation')
    results.to_csv('{}.csv'.format(outpath), index=False)