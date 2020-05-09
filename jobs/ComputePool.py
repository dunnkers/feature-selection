#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import pandas as pd
import numpy as np
from util import get_datasets, get_poolSize, get_outpath

def ranking_pool(_, filename, ranking_func):
    # Load data
    test_data = pd.read_csv(filename, sep='\t')
    # Separate features/target
    X = test_data.drop('Class', axis=1).values
    y = test_data['Class'].values

    # Execute feature ranking
    start = time.time()
    ranking = ranking_func(X, y)
    elapsed = time.time() - start
    print('FS {} took {} seconds.'.format(ranking_func.__name__, elapsed))

    # Write results to CSV
    feature_names = test_data.drop('Class', axis=1).columns.values
    output = pd.DataFrame(index=feature_names, data={\
        'FeatureRank': ranking,\
        'Time': elapsed\
    })
    return output

def run_ranking_pool(ranking_func):
    run_pool(ranking_pool, 'ranking', ranking_func)

def run_pool(func, outfolder, map_args=[]):
    begin = time.time()
    dataset_dir = sys.argv[1]
    inputs = sorted(get_datasets(dataset_dir)) # sort datasets by filename
    poolSize = get_poolSize()

    # start pool
    print('Starting pool of size {} with {} inputs'.format(poolSize, len(inputs)))
    pool = multiprocessing.Pool(processes=poolSize,)
    pool_args = map(lambda x: np.append(x, map_args),\
        enumerate(inputs, start=1)) # construct tuple as input parameters
    poolResults = pool.starmap(func, pool_args)

    # collect results
    pool.close() # Stop pool
    pool.join()  # Wrap up data from workers
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , poolSize, ' workers: ', elapsedTime, ' seconds')

    # concatenate results. remove duplicate columns
    results_keys = range(1, len(poolResults)+1) # numbering according to filename
    results = pd.concat(poolResults, axis=1, keys=results_keys)

    # write to csv
    outpath = get_outpath(dataset_dir, outfolder)
    results.to_csv('{}.csv'.format(outpath))