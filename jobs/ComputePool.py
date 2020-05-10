#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import pandas as pd
import numpy as np
from util import get_datasets, get_poolSize, get_outpath

def ranking_pool(dataset_no, filename, ranking_func, ranking_method):
    print(ranking_method)

    # Load data
    test_data = pd.read_csv(filename, sep='\t')
    # Separate features/target
    X = test_data.drop('Class', axis=1).values
    y = test_data['Class'].values

    # Execute feature ranking
    start = time.time()
    ranking = ranking_func(X, y)
    elapsed = time.time() - start
    print('FS {} took {} seconds.'.format(ranking_method, elapsed))

    # Write results to CSV
    feature_names = test_data.drop('Class', axis=1).columns.values
    output = pd.DataFrame(data={
        'FeatureName': feature_names,
        'FeatureRank': ranking,
        'Time': elapsed,
        'DatasetNo': dataset_no,
        'RankingMethod': ranking_method
    })
    return output

def run_ranking_pool(ranking_func, ranking_method=None):
    run_pool(ranking_pool, 'ranking', ranking_func,\
        ranking_method or ranking_func.__name__)

def run_pool(func, outfolder, *map_args):
    dataset_dir = sys.argv[1]
    inputs = sorted(get_datasets(dataset_dir)) # sort datasets by filename
    poolSize = get_poolSize()

    # configure pool
    inputs = [inputs[0], inputs[1]]
    pool = multiprocessing.Pool(processes=poolSize,)
    pool_args = map(lambda x: np.append(x, map_args),\
        enumerate(inputs, start=1)) # construct tuple as input parameters

    # run pool
    print('Running pool... [{} workers, {} datasets]'.format(\
        poolSize, len(inputs)))
    begin = time.time()
    pool_results = pool.starmap(func, pool_args)
    pool.close() # Stop pool
    pool.join()  # Wrap up data from workers
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , poolSize, ' workers: ', elapsedTime, ' seconds')

    # concatenate results. remove duplicate columns
    results = pd.concat(pool_results)

    # write to csv
    outpath = get_outpath(dataset_dir, outfolder)
    results.to_csv('{}.csv'.format(outpath), index=False)