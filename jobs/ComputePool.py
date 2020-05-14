#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import math
import pandas as pd
import numpy as np
import os

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

def ranking_pool(filename, job_id, ranking_func, ranking_method):
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
    outpath = '{}.csv'.format(\
        os.path.join('results', job_id, 'ranking', filename))
    save_output(output, outpath)

def run_ranking_pool(ranking_func, ranking_method=None):
    ranking_method = ranking_method or ranking_func.__name__
    print('Ranking pool [{}]'.format(ranking_method))
    run_pool(ranking_pool, ranking_func, ranking_method)

def validation_pool(filename, job_id, validation_func, validation_method):
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
    outpath = '{}.csv'.format(\
        os.path.join('results', job_id, 'validation', filename))
    save_output(results, outpath)

def run_validation_pool(validation_func, validation_method=None):
    validation_method = validation_method or validation_func.__name__
    print('Validation pool [{}]'.format(validation_method))
    run_pool(validation_pool, validation_func, validation_method)

def run_pool(func, *map_args):
    data_collection, job_id, task_id = sys.argv[1], sys.argv[2], sys.argv[3]
    data_col = pd.read_csv(data_collection)
    n_rows, _ = data_col.shape
    pool_size = multiprocessing.cpu_count()
    tasks = math.ceil(n_rows / pool_size)
    # inputs for this task
    inputs = np.array_split(data_col.path, tasks)[int(task_id)]
    inputs = inputs.map(lambda input: [input, job_id, *map_args])
    print('Running pool... [{} workers, {} datasets, {} tasks]'.format(\
        pool_size, len(inputs), tasks))
    begin = time.time()
    pool = multiprocessing.Pool(processes=pool_size,)
    pool.starmap(func, inputs)
    pool.close() # Stop pool
    pool.join()  # Wrap up data from workers
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , pool_size, ' workers: ', elapsedTime, ' seconds')