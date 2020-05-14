#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import math
import pandas as pd
import numpy as np
import os
from util import get_datasets, get_poolSize, get_outpath

def load_dataset(filename):
    dataset = pd.read_csv(filename, sep='\t')
    X = dataset.drop('Class', axis=1).values
    y = dataset['Class'].values
    return (dataset, X, y)

def save_output(output, outpath):
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.exists(outpath): # merge with existing results
        output = pd.concat([ pd.read_csv(outpath), output ])
    output.to_csv(outpath, index=False)

def ranking_pool(filename, ranking_func, ranking_method):
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
        'ranking_method': ranking_method
    })

    # Save to .csv
    outpath = '{}.csv'.format(os.path.join('results', 'ranking', filename))
    save_output(output, outpath)

def run_ranking_pool(ranking_func, ranking_method=None):
    ranking_method = ranking_method or ranking_func.__name__
    print('Ranking pool [{}]'.format(ranking_method))
    run_pool(ranking_pool, 'ranking', ranking_func, ranking_method)

def validation_pool(filename, validation_func, validation_method):
    dataset, _, y = load_dataset(filename)

    # Load feature ranking
    inpath = '{}.csv'.format(os.path.join('results', 'ranking', filename))
    data = pd.read_csv(inpath)

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
            
    results['validation_method'] = validation_method
    outpath = '{}.csv'.format(os.path.join('results', 'validation', filename))
    save_output(results, outpath)

def run_validation_pool(validation_func, validation_method=None):
    validation_method = validation_method or validation_func.__name__
    print('Validation pool [{}]'.format(validation_method))
    run_pool(validation_pool, 'validation', validation_func, validation_method)

def run_pool(func, outfolder, *map_args):
    _, data_collection, job_id, task_id = sys.argv
    data_col = pd.read_csv(data_collection)
    n_rows, _ = data_col.shape
    pool_size = multiprocessing.cpu_count()
    tasks = math.ceil(n_rows / pool_size)
    # inputs for this task
    inputs = np.array_split(data_col.path, tasks)[int(task_id)]
    inputs = inputs.map(lambda input: np.append(input, map_args))
    print('Running pool... [{} workers, {} datasets, {} tasks]'.format(\
        pool_size, len(inputs), tasks))
    begin = time.time()
    pool = multiprocessing.Pool(processes=pool_size,)
    pool.starmap(func, inputs)
    pool.close() # Stop pool
    pool.join()  # Wrap up data from workers
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , pool_size, ' workers: ', elapsedTime, ' seconds')


    # 1. put all replicates in data descriptor
    # 2. make ranking/validation functions 1:1;
        # - save results in exact path
        # - remove dataset_no

    # read_csv ... etcetera.
    # pd.read_csv(data_collection).loc[int(task_id)]
    exit()




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