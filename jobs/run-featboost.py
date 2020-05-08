#!/usr/bin/env python
from ComputePool import run_pool
from FeatBoost import run_FeatBoost

if __name__ == '__main__':
    print('Starting FeatBoost feature selection ranking pool.')
    run_pool(run_FeatBoost, 'ranking')
