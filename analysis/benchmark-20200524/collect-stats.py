import pandas as pd
rankdata = pd.read_csv('./analysis/benchmark-20200524/data/rankdata.csv')
rankdata.head()

from tqdm import tqdm
tqdm.pandas()

import numpy as np
from sklearn.utils import Bunch


# def add_stats(row):
#     if pd.isna(row.p_informative):
#         return row
#     subset_rows = rankdata[(rankdata['description'] == row.description) &\
#                            (rankdata['ranking_method'] == row.ranking_method) &\
#                            (rankdata['replica_no'] == row.replica_no) &\
#                            (rankdata['feature_rank'] <= row.feature_rank)]
#     subset = subset_rows['feature_index'].values
#     row['subset'] = subset
    # p_informative = [int(x) for x in row.p_informative.split(',')]
    # row['n_positives'] = len(subset)
    # row['n_true_positives'] = np.sum(np.isin(subset, p_informative))
    # row['n_false_positives'] = row['n_positives'] - row['n_true_positives']
    # row['n_false_negatives'] = len(p_informative) - row['n_true_positives']
    # row['recall'] = row['n_true_positives'] / len(p_informative)
    # row['precision'] = row['n_true_positives'] / row['n_positives']
#     return row
# statdata = rankdata.progress_apply(add_stats, axis=1)

rankdata=rankdata[rankdata['name']=='XOR Models']
# index dataframe
indexed=rankdata.set_index(['description', 'ranking_method', 'replica_no'])
indexed=indexed.sort_index()
def construct_subsets(row):
    description, ranking_method, replica_no = row.name
    run = indexed.loc[(description, ranking_method, replica_no)]
    subset_rows = run[run.feature_rank <= row.feature_rank]
    subset = subset_rows.feature_index.values
    row['subset'] = subset
    return row
# indexed['subset'] = indexed['feature_rank'].map(construct_subsets)
indexed = indexed.progress_apply(construct_subsets, axis=1)
indexed.to_csv('./data/indexed.csv')


# Decode p_informative with 
from ast import literal_eval




# 
statdata = pd.DataFrame() # data, statistically post-processed
group_keys = ['description', 'ranking_method', 'replica_no']
grouped = rankdata.groupby(group_keys)
t = tqdm(grouped)
for keys, rankset in t:
    description, ranking_method, replica_no = keys
    t.set_description(description + ', ' + ranking_method + ', ' + str(replica_no))
    p_informatives = np.unique(rankset['p_informative'].values) # takes long for NaN's..
    if pd.isna(p_informatives).any(): # no `p_informative` set
        continue

    p_informative = [int(x) for x in p_informatives[0].split(',')]
    for feature_rank, ranks in rankset.groupby('feature_rank'):
        subset_rows = rankset[rankset['feature_rank'] <= feature_rank]
        subset = subset_rows['feature_index'].values

        stat = Bunch()
        stat.n_positives = len(subset)
        stat.n_true_positives = np.sum(np.isin(subset, p_informative))
        stat.n_false_positives = stat.n_positives - stat.n_true_positives
        stat.n_false_negatives = len(p_informative) - stat.n_true_positives
        stat.recall = stat.n_true_positives / len(p_informative)
        stat.precision = stat.n_true_positives / stat.n_positives
        statdata = statdata.append(ranks.assign(**stat))
    # break
statdata.to_csv('./data/statdata.csv', index=False)
statdata.shape