#!/usr/bin/env python
import multiprocessing 
import time
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

def ranking_pool(ranking_func, ranking_method, filename, job_id):
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

def validation_pool(validation_func, validation_method, filename, job_id):
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

def run_pool(func, *map_args):
    # Determine task division according to arguments
    data_col_path, job_id, task_id = sys.argv[1], sys.argv[2], sys.argv[3]
    data_collection = pd.read_csv(data_col_path)
    n_datasets, _ = data_collection.shape
    pool_size = multiprocessing.cpu_count()
    tasks = math.ceil(n_datasets / pool_size)

    # Prepare inputs
    inputs = np.array_split(data_collection.path, tasks)[int(task_id)]
    inputs = inputs.map(lambda input: [*map_args, input, job_id])

    # Run pool
    print('Running pool `{}` task {}... [{} workers, {} datasets, {} tasks]'.format(
        job_id, task_id, pool_size, len(inputs), tasks))
    print('\tmap_args = {}'.format(map_args))
    begin = time.time()
    pool = multiprocessing.Pool(processes=pool_size,)
    pool.starmap(func, inputs)
    pool.close() # Stop pool
    pool.join()  # Wrap up data from workers
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , pool_size, ' workers: ', elapsedTime, ' seconds')