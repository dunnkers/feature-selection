import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import sys
from fseval.pipeline.compute_pool import save_output
tqdm.pandas()

def add_groundtruth_stats(rankdata):
    statdata = rankdata.dropna(subset=['p_informative'])
    if statdata.empty: # no ground truth
        return statdata

    statdata['subset' ]          = statdata['subset'].str.split(pat=',')
    statdata['p_informative']    = statdata['p_informative'].astype(str)\
                                                     .str.split(pat=',')
    n_p_informative              = statdata['p_informative'].str.len()
    n_positives                  = statdata['subset'].str.len()

    compute_tp = lambda x: np.sum(np.isin(x.subset, x.p_informative))
    tp = statdata.apply(compute_tp, axis='columns')

    fp                      = n_positives - tp
    # fn          = n_p_informative - tp
    tn                      = statdata['p'].astype(int) - n_positives
    statdata['recall']      = tp / n_p_informative
    statdata['precision']   = tp / n_positives
    statdata['fpr']         = fp / (fp + tn)
    statdata['f1']          = 2 * ((statdata['precision'] * statdata['recall'])\
                                /  (statdata['precision'] + statdata['recall']))

    compute_auc = lambda x: np.trapz([0, x.recall, 1], [0, x.fpr, 1])
    statdata['auc'] = statdata.apply(compute_auc, axis='columns')

    return statdata

def collect_results(datacol, transformers, *result_dirs):
    t = tqdm(datacol.iterrows(), total=len(datacol))
    for _, dataset in t:
        t.set_description('{} {}'.format(
            dataset.description, dataset.replica_no))

        data = pd.DataFrame()
        outdir = None
        for result_dir in result_dirs:
            rankpath = os.path.join(result_dir, 'ranking', dataset.path + '.csv')
            if os.path.exists(rankpath):
                # overwrite last data, always use last ranking
                data = pd.read_csv(rankpath)
                outdir = result_dir

        if data.empty:
            continue
        outpath = os.path.join(outdir, 'stats', dataset.path + '.csv')

        dataset = dataset.drop(labels=[ # `labels` cause `dataset` is Series
            'directory', 'path', 'adapter']) # drop some cols to save space
        # add more specific info to the ranking rows; e.g. replica_no, etc.
        rankdata = data.assign(**dataset)

        # add statistics, etc. whatever transformations we want.
        for transformer in transformers:
            rankdata = transformer(rankdata)

        outdir = os.path.dirname(outpath)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        rankdata = rankdata.drop(columns=[ # storage space reduction
            'subset', 'replicas', 'feature_index', 'p', 'n', 'p_informative'])
        rankdata.to_csv(outpath, index=False)

import pandas as pd
datacol = pd.read_csv('./data/10k_features.csv')
collect_results(datacol, [ add_groundtruth_stats ],
    './results/11714894_subsets',
    './results/11824789')
print('statdata successfully exported.')