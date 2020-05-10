#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import pandas as pd
import numpy as np
import os
from util import get_datasets, get_poolSize, get_outpath

def load_dataset(filename):
    dataset = pd.read_csv(filename, sep='\t')
    X = dataset.drop('Class', axis=1).values
    y = dataset['Class'].values
    return (dataset, X, y)

def ranking_pool(dataset_no, filename, ranking_func, ranking_method):
    dataset, X, y = load_dataset(filename)

    # Execute feature ranking
    start = time.time()
    ranking = ranking_func(X, y)
    time_elapsed = time.time() - start
    print('FS {} took {} seconds.'.format(ranking_method, time_elapsed))

    # Write results to CSV
    feature_names = dataset.drop('Class', axis=1).columns.values
    output = pd.DataFrame(data={
        'feature_name': feature_names,
        'feature_rank': ranking,
        'time_elapsed': time_elapsed,
        'dataset_no': dataset_no,
        'ranking_method': ranking_method
    })
    return output

def run_ranking_pool(ranking_func, ranking_method=None):
    ranking_method = ranking_method or ranking_func.__name__
    print('Running {} ranking pool...'.format(ranking_method))
    run_pool(ranking_pool, 'ranking', ranking_func, ranking_method)

def validation_pool(dataset_no, filename, validation_func, validation_method):
    dataset, _, y = load_dataset(filename)

    # Load feature ranking
    dataset_dir = os.path.dirname(filename)
    inpath = '{}.csv'.format(get_outpath(dataset_dir, 'ranking'))
    results_data = pd.read_csv(inpath)
    current = results_data.loc[results_data['dataset_no'] == int(dataset_no)]
    # ranking = current['feature_rank'].dropna().sort_values().index
    ranking = current.sort_values('feature_rank')['feature_name']\
        .dropna().values
    # -> e.g. ['feature_name', 'another_feature', ...] in order of feature rank

    # Construct subsets
    subsets = map(lambda i: (ranking[i], ranking[0:i+1]), range(len(ranking)))
    # -> e.g. [[<best ranked feature>], [<top 2 ranked features>], ... etc]

    results = pd.DataFrame()
    for index, subset in subsets:
        X_subset = dataset[subset].values

        # run validation
        start = time.time()
        scores = validation_func(X_subset, y)
        time_elapsed = time.time() - start

        # store results
        scores['feature_name'] = index
        scores['dataset_no'] = dataset_no
        scores['time_elapsed'] = time_elapsed
        scores['validation_method'] = validation_method
        results = results.append(scores)

    return results

def run_validation_pool(validation_func, validation_method=None):
    validation_method = validation_method or validation_func.__name__
    print('Running {} validation pool...'.format(validation_method))
    run_pool(validation_pool, 'validation', validation_func, validation_method)

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