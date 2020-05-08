#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import pandas as pd
from util import get_datasets, get_poolSize, get_outpath

def run_pool(func, outfolder):
    begin = time.time()
    dataset_dir = sys.argv[1]
    inputs = sorted(get_datasets(dataset_dir)) # sort datasets by filename
    poolSize = get_poolSize()

    # start pool
    pool = multiprocessing.Pool(processes=poolSize,)
    inputs = [inputs[0]]
    poolResults = pool.map(func, enumerate(inputs, start=1))

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