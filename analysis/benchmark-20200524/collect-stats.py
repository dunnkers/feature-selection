import pandas as pd
import numpy as np
import time
rankdata = pd.read_csv('./analysis/benchmark-20200524/data/rankdata.csv')
print('computing subset...')
start = time.time()
rankdata['subset'] = rankdata['subset'].str.split(pat=',')
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing p_informative...')
start = time.time()
rankdata['p_informative'] = rankdata['p_informative'].str.split(pat=',')
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing total_positives...')
start = time.time()
rankdata['total_positives'] = rankdata['p_informative'].str.len()
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing n_positives...')
start = time.time()
rankdata['n_positives'] = rankdata['subset'].str.len()
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing tp...')
start = time.time()
rankdata['tp'] = rankdata['subset'].combine(rankdata['p_informative'], np.isin)
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing tp np.sum...')
start = time.time()
rankdata['tp'] = rankdata['tp'].map(np.sum)
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing fp...')
start = time.time()
rankdata['fp'] = rankdata['n_positives'] - rankdata['tp']
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing fn...')
start = time.time()
rankdata['fn'] = rankdata['total_positives'] - rankdata['tp']
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing recall...')
start = time.time()
rankdata['recall'] = rankdata['tp'] / rankdata['total_positives']
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing precision...')
start = time.time()
rankdata['precision'] = rankdata['tp'] / rankdata['n_positives']
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

print('computing f1 score...')
start = time.time()
rankdata['f1'] = 2 * ((rankdata['precision'] * rankdata['recall']) / \
                      (rankdata['precision'] + rankdata['recall']))
elapsed = time.time() - start
print('\ttook {} sec'.format(elapsed))

# 💡
# another idea to compute this, is to treat `subset` as an array full of True,
# and `p_informative` as an array of the same shape, but partially True values.

# rankdata['subset'] = rankdata['subset']\
#                         .map(lambda s: s.split(','))
# rankdata['n_positives'] = rankdata['subset']\
#                         .map(lambda s: len(s))
# rankdata['subset'] = rankdata['subset'].str.split(pat=',')
# rankdata['p_informative'] = rankdata['p_informative']\
#                         .map(lambda s: [] if pd.isna(s) else s.split(','))

rankdata.to_csv('./data/statdata.csv', index=False)
print(rankdata.shape)


# compute it all


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