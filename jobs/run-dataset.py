#!/usr/bin/env python
 
import multiprocessing 
import os # For reading the number of CPUs requested. 
import time # For clocking the calculation. 
import sys
import pandas as pd
from RFE import runRFE

def get_datasets(dataset_dir):
    dataset_files = []
    for dirpath, _, filenames in os.walk('{}'.format(dataset_dir)):
        # In some directories, there are report files. Skip these files.
        if len(filenames) < 10 and len(dirpath) > 0:
            continue

        for filename in filenames:
            if not filename.endswith('.txt.gz'):
                continue
            dataset_file = os.path.join(dirpath, filename)
            dataset_files.append(dataset_file)
    return dataset_files

def get_poolSize():
    key = 'SLURM_JOB_CPUS_PER_NODE' # Requested CPUs.
    local = multiprocessing.cpu_count() # Local available CPU's
    return int(os.environ(key) if os.environ.get(key) is not None else local)

if __name__ == '__main__':
    begin = time.time()
    dataset_dir = sys.argv[1]
    inputs = get_datasets(dataset_dir)
    poolSize = get_poolSize()

    # start pool
    pool = multiprocessing.Pool(processes=poolSize,)
    poolResults = pool.map(runRFE, inputs) # Do the calculation.

    # collect results
    pool.close() # Stop pool accordingly.
    pool.join()  # Wrap up data from the workers in the pool.
    results = pd.concat(poolResults, axis=1)
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , poolSize, ' workers: ', elapsedTime, ' seconds')

    # output path - uses same directory structure as original /data folder
    outpath = os.path.relpath(dataset_dir, os.path.join(os.getcwd(), 'data'))
    results.to_csv('results/{}.csv'.format(outpath), index=False)