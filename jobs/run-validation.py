#!/usr/bin/env python
from ComputePool import run_pool
from Validation import run_validation

if __name__ == '__main__':
    print('Starting validation pool.')
    run_pool(run_validation, 'validation')
