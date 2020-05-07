import os
from multiprocessing import cpu_count
from os import path, walk, environ, getcwd, makedirs

def getDatasetNo(filename):
    return filename.split('_')[-1:][0].split('.')[0]

def get_datasets(dataset_dir):
    dataset_files = []
    for dirpath, _, filenames in walk('{}'.format(dataset_dir)):
        # In some directories, there are report files. Skip these files.
        if len(filenames) < 10 and len(dirpath) > 0:
            continue

        for filename in filenames:
            if not filename.endswith('.txt.gz'):
                continue
            dataset_file = path.join(dirpath, filename)
            dataset_files.append(dataset_file)
    return dataset_files

def get_outpath(dataset_dir, out_dir):
    # using same directory structure as original /data folder
    dataset_path = path.relpath(dataset_dir, path.join(getcwd(), 'data'))
    outpath = path.join(out_dir, dataset_path)
    if not path.isdir(path.dirname(outpath)):
        makedirs(path.dirname(outpath))
    return outpath

def get_poolSize():
    key = 'SLURM_JOB_CPUS_PER_NODE' # Requested CPUs.
    local = cpu_count() # Local available CPU's
    return int(environ[key] if environ.get(key) is not None else local)