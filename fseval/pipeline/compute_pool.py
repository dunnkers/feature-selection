#!/usr/bin/env python
import multiprocessing 
import time
import sys
import math
import pandas as pd
import numpy as np
import os
import importlib
from datetime import datetime
import argparse

def load_dataset(datacol):
    path = '...dataset_adapters.{}'.format(datacol.adapter)
    adapter = importlib.import_module(path, __name__)
    dataset = adapter.load_dataset(datacol.path)
    return dataset

def save_output(output, outpath):
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.exists(outpath): # merge with existing results
        output = pd.concat([ pd.read_csv(outpath), output ])
    output.to_csv(outpath, index=False)

def ranking_pool(ranking_func, ranking_method, datacol, batch_id):
    filename = datacol.path
    dataset = load_dataset(datacol)

    # Execute feature ranking
    start = time.time()
    ranking_all = ranking_func(dataset.data, dataset.target)
    time_elapsed = time.time() - start
    print('{} feature selection took {} seconds.'.format(\
        ranking_method, time_elapsed))

    # Write results to .csv
    ranking = pd.Series(ranking_all).dropna()
    subsets = ranking.map(lambda rank: ','.join(\
            ranking[ranking <= rank].index.astype(str)))
    output = pd.DataFrame(data={
        'feature_index': ranking.index,
        'feature_rank': ranking.values,
        'subset': subsets.values,
        'cpu_time': time_elapsed,
        'ranking_method': ranking_method
    })

    # Save to .csv
    outpath = '{}.csv'.format(\
        os.path.join('results', batch_id, 'ranking', filename))
    save_output(output, outpath)

def validation_pool(validation_func, validation_method, datacol, batch_id):
    filename = datacol.path
    dataset = load_dataset(datacol)

    # Load feature ranking
    inpath = '{}.csv'.format(os.path.join('results', batch_id, 'ranking', filename))
    data = pd.read_csv(inpath)

    # Split by each ranking method
    results = pd.DataFrame()
    for ranking_method, ranking_data in data.groupby('ranking_method'):
        print('validating {}'.format(ranking_method))

        # iterate ranked feature groups. `groupby` sorts by feature rank.
        for feature_rank, _ in ranking_data.groupby('feature_rank'):
            print('validating subset with rank {}'.format(feature_rank))

            # construct subset- use all features up to this rank.
            subset = ranking_data[ranking_data['feature_rank'] <= feature_rank]
            subset_indices = subset['feature_index'].values
            X_subset = np.take(dataset.data, subset_indices, axis=1)

            # run validation
            start = time.time()
            scores = validation_func(X_subset, dataset.target)
            time_elapsed = time.time() - start

            # store results
            scores['n_features'] = np.size(X_subset, axis=1)
            scores['ranking_method'] = ranking_method
            scores['cpu_time'] = time_elapsed
            results = results.append(scores)
            
    results['validation_method'] = validation_method
    outpath = '{}.csv'.format(\
        os.path.join('results', batch_id, 'validation', filename))
    save_output(results, outpath)

def get_cpu_count():
    key = 'SLURM_JOB_CPUS_PER_NODE'     # requested CPUs in SLURM job
    return int(os.environ.get(key, multiprocessing.cpu_count()))

def run_pool(func, *map_args):
    # Parse arguments
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    parser = argparse.ArgumentParser(description='Run feature selection pool.')
    parser.add_argument('data_col_path')
    parser.add_argument('--batch_id', default=timestamp)
    parser.add_argument('--job_id', default='0')
    parser.add_argument('--task_id', default='0')
    args = parser.parse_args()

    # Determine task division according to arguments
    print('... reading data collection from: {}'.format(args.data_col_path))
    data_collection = pd.read_csv(args.data_col_path)
    n_datasets, _ = data_collection.shape
    cpus = get_cpu_count() # controlled by SLURM
    cpus_per_task = 1
    pool_size = math.ceil(cpus / cpus_per_task)
    n_tasks = math.ceil(n_datasets / pool_size)

    # Prepare inputs
    datasets = np.array_split(data_collection, n_tasks)[int(args.task_id)]
    # set more meaningful job description
    descrs = np.unique(datasets.get('description', datasets.get('path', [])))
    descr = descrs[0] if len(descrs) == 1 else 'task {}/{} w/ {} datasets'\
        .format(args.task_id, n_tasks, len(descrs))
    os.system('scontrol update JobID={} name="{}"'.format(args.job_id, descr))
    inputs = map(lambda row: [*map_args, row[1], args.batch_id], datasets.iterrows())

    # Run pool
    print('''Running pool `{}` with pool size {} task {}... \
        [{} workers, {} datasets, {} tasks]'''.format(
        args.batch_id, pool_size, args.task_id,\
        cpus, len(datasets), n_tasks))
    print('\tmap_args = {}'.format(map_args))
    print('\tdatasets:')
    print('\t\t', *(enumerate(descrs)), sep='\n\t\t')
    begin = time.time()
    pool = multiprocessing.Pool(processes=pool_size,)
    pool.starmap(func, inputs)
    pool.close()
    pool.join()
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , pool_size, ' workers: ', elapsedTime, ' seconds')