import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import sys

def add_groundtruth_stats(rankdata):
    statdata = rankdata.dropna(subset=['p_informative'])
    statdata['subsets'] = statdata['subset'].str.split(pat=',')
    statdata['p_informatives'] = statdata['p_informative'].str.split(pat=',')
    statdata['n_p_informatives'] = statdata['p_informatives'].str.len()
    statdata['n_positives'] = statdata['subsets'].str.len()

    compute_tp = lambda x: np.sum(np.isin(x.subsets, x.p_informatives))
    statdata['tp'] = statdata.apply(compute_tp, axis='columns')

    statdata['fp']          = statdata['n_positives'] - statdata['tp']
    statdata['fn']          = statdata['n_p_informatives'] - statdata['tp']
    statdata['tn']          = statdata['p'].astype(int) - statdata['n_positives']
    statdata['recall']      = statdata['tp'] / statdata['n_p_informatives']
    statdata['precision']   = statdata['tp'] / statdata['n_positives']
    statdata['fpr']         = statdata['fp'] / (statdata['fp'] + statdata['tn'])
    statdata['f1']          = 2 * ((statdata['precision'] * statdata['recall'])\
                                /  (statdata['precision'] + statdata['recall']))

    compute_auc = lambda x: np.trapz([0, x.recall, 1], [0, x.fpr, 1])
    statdata['auc'] = statdata.apply(compute_auc, axis='columns')

    return statdata

def collect_results(datacol, transformers, *ranking_dirs):
    statdata = pd.DataFrame()
    chunks = []
    t = tqdm(datacol.iterrows(), total=len(datacol), position=0)
    for _, dataset in t:
        t.set_description('{} {} [statdata size = {}]'.format(
            dataset.description, dataset.replica_no, sys.getsizeof(chunks)
        ))
        data = pd.DataFrame()
        for ranking_dir in ranking_dirs:
            rankpath = os.path.join(ranking_dir, dataset.path + '.csv')
            if os.path.exists(rankpath):
                # overwrite last data, always use last ranking
                data = pd.read_csv(rankpath)

        if data.empty:
            continue

        dataset = dataset.drop(labels=[\
            'directory', 'path', 'adapter']) # drop some cols to save space
        # add more specific info to the ranking rows; e.g. replica_no, etc.
        rankdata = data.assign(**dataset)

        # add statistics, etc. whatever transformations we want.
        for transformer in transformers:
            rankdata = transformer(rankdata)
        
        chunks.append(rankdata)
    statdata = pd.concat(chunks)
    del chunks
    return statdata