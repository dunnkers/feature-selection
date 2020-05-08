#!/usr/bin/env python
 
import multiprocessing 
import time # For clocking the calculation. 
import sys
import pandas as pd
from RFE import runRFE
from util import get_datasets, get_poolSize, get_outpath

if __name__ == '__main__':
    begin = time.time()
    dataset_dir = sys.argv[1]
    inputs = get_datasets(dataset_dir)
    poolSize = get_poolSize()

    # start pool
    pool = multiprocessing.Pool(processes=poolSize,)
    poolResults = pool.map(runRFE, [inputs[0],inputs[1],inputs[2]]) # Do the calculation.

    # collect results
    pool.close() # Stop pool accordingly.
    pool.join()  # Wrap up data from the workers in the pool.
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , poolSize, ' workers: ', elapsedTime, ' seconds')

    # concatenate results. remove duplicate columns
    poolResults = map(lambda df: df.drop('FeatureName', axis=1), poolResults)
    results = pd.concat(poolResults, axis=1)

    # write to csv
    outpath = get_outpath(dataset_dir, 'ranking')
    results.to_csv('{}.csv'.format(outpath), index=False)