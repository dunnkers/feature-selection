#!/usr/bin/env python
from ComputePool import run_pool
from RFE import run_RFE

if __name__ == '__main__':
    print('Starting RFE feature selection ranking pool.')
    run_pool(run_RFE, 'ranking')
