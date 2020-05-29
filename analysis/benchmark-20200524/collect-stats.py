### Add 'subset' fields to results
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
tqdm.pandas()

def save_output(output, outpath):
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output.to_csv(outpath, index=False)

datacol = pd.read_csv('./analysis/benchmark-20200524/data/benchmark_24-replicas.csv')
t = tqdm(datacol.iterrows(), total=len(datacol))
for _, dataset in t:
    rankpath = os.path.join('./results/11714894/ranking', dataset.path + '.csv')
    if not os.path.exists(rankpath):
        continue
    t.set_description(dataset.description)
    data = pd.read_csv(rankpath)
    for ranking_method, groupeddata in data.groupby('ranking_method'):
        assert(len(np.unique(groupeddata.cpu_time)) == 1) # if not ranking gets messed up

        subsets = groupeddata.apply(lambda x: ','.join(\
            groupeddata[groupeddata['feature_rank'] <= x['feature_rank']]\
            ['feature_index'].astype(str)), axis=1)
        data.loc[subsets.index, 'subset'] = subsets

    save_output(data, os.path.join('./results/11714894_subsets/ranking',\
                            dataset.path + '.csv'))

# rankdata = pd.read_csv('./analysis/benchmark-20200524/data/rankdata.csv')
# from sklearn.utils import Bunch
# # 
# statdata = pd.DataFrame() # data, statistically post-processed
# group_keys = ['description', 'ranking_method', 'replica_no']
# grouped = rankdata.groupby(group_keys)
# t = tqdm(grouped)
# for keys, rankset in t:
#     description, ranking_method, replica_no = keys
#     t.set_description(description + ', ' + ranking_method + ', ' + str(replica_no))
#     p_informatives = np.unique(rankset['p_informative'].values) # takes long for NaN's..
#     if pd.isna(p_informatives).any(): # no `p_informative` set
#         continue

#     p_informative = [int(x) for x in p_informatives[0].split(',')]
#     for feature_rank, ranks in rankset.groupby('feature_rank'):
#         subset_rows = rankset[rankset['feature_rank'] <= feature_rank]
#         subset = subset_rows['feature_index'].values

#         stat = Bunch()
#         stat.n_positives = len(subset)
#         stat.n_true_positives = np.sum(np.isin(subset, p_informative))
#         stat.n_false_positives = stat.n_positives - stat.n_true_positives
#         stat.n_false_negatives = len(p_informative) - stat.n_true_positives
#         stat.recall = stat.n_true_positives / len(p_informative)
#         stat.precision = stat.n_true_positives / stat.n_positives
#         statdata = statdata.append(ranks.assign(**stat))
#     # break
# statdata.to_csv('./data/statdata.csv', index=False)
# statdata.shape