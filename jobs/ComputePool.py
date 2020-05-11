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
    ranking = pd.Series(ranking).dropna() # don't store unranked features
    feature_names = np.take(feature_names, ranking.index) # filter accordingly
    output = pd.DataFrame(data={
        'feature_name': feature_names,
        'feature_rank': ranking,
        'cpu_time': time_elapsed,
        'dataset_no': dataset_no,
        'ranking_method': ranking_method
    })
    return output

def run_ranking_pool(ranking_func, ranking_method=None):
    ranking_method = ranking_method or ranking_func.__name__
    print('Ranking pool [{}]'.format(ranking_method))
    run_pool(ranking_pool, 'ranking', ranking_func, ranking_method)

def validation_pool(dataset_no, filename, validation_func, validation_method):
    dataset, _, y = load_dataset(filename)

    # Load feature ranking
    dataset_dir = os.path.dirname(filename)
    inpath = '{}.csv'.format(get_outpath(dataset_dir, 'ranking'))
    all_data = pd.read_csv(inpath)
    data = all_data.loc[all_data['dataset_no'] == int(dataset_no)]

    # Split by each ranking method
    results = pd.DataFrame()
    for ranking_method, ranking_data in data.groupby('ranking_method'):
        ranking = ranking_data.sort_values('feature_rank')['feature_name']\
            .dropna().values
        
        # Loop all result rows
        for _, item in ranking_data.iterrows():
            subset = ranking[:int(item.feature_rank)] # all up till this feature
            X_subset = dataset[subset].values

            # run validation
            start = time.time()
            scores = validation_func(X_subset, y)
            time_elapsed = time.time() - start

            # store results
            scores['n_features'] = len(subset)
            scores['ranking_method'] = ranking_method
            scores['cpu_time'] = time_elapsed
            results = results.append(scores)
            
    results['dataset_no'] = dataset_no
    results['validation_method'] = validation_method
    return results

def run_validation_pool(validation_func, validation_method=None):
    validation_method = validation_method or validation_func.__name__
    print('Validation pool [{}]'.format(validation_method))
    run_pool(validation_pool, 'validation', validation_func, validation_method)

def run_pool(func, outfolder, *map_args):
    dataset_dir = sys.argv[1]
    inputs = sorted(get_datasets(dataset_dir)) # sort datasets by filename
    poolSize = get_poolSize()

    # configure pool
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
    outpath = '{}.csv'.format(get_outpath(dataset_dir, outfolder))
    if os.path.exists(outpath): # merge with existing results
        results = pd.concat([ pd.read_csv(outpath), results ])
    results.to_csv(outpath, index=False)