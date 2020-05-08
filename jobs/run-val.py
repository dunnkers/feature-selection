#!/usr/bin/env python

from KNN import runKNN
from ComputePool import run_pool

if __name__ == '__main__':
    print('Starting validation pool.')
    run_pool(runKNN, 'validation')
