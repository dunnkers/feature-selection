### Adds more info to a data collection descriptor file (e.g. `rebate.csv`),
### such as amount of features `p`, amount of samples `n`, or dataset name.

import pandas as pd
import os
import numpy as np

all = pd.read_csv('./data/benchmark_24-replicas.csv')
info = pd.read_csv('./data/rebate/rebate-datasets.csv')
all2 = all.drop(['p', 'n', 'name'], axis=1)

# make sure we're using dot decimal separator
import locale; locale.setlocale(locale.LC_ALL, ""); print(locale.localeconv()["decimal_point"])

### Construct a `description`; a more informative title for the dataset
def doit(row):
    name = row['name']
    p = row['p']
    n = row['n']
    h = row['heritability']
    description = '{} [p={},n={},h={}'.format(name, p, n,\
        (h + '').replace(',', '.'))

    edm = row['edm']
    if not edm == 'None':
        description = '{},edm={}'.format(description, edm)

    p_alt = row['p_alternate']
    if not p_alt == 'None':
        description = '{},p_alt={}'.format(description, p_alt)

    row['description'] = description + ']'

    return row
info2 = info.apply(doit, axis=1)

import csv
# translate p_key feature names to feature indices
for index, dataset in info2.iterrows():
    files = os.listdir(dataset.directory)
    file = files[0]
    # fpath = os.path.join(dataset.directory, file)
    # with open(fpath, 'r') as infile:
    #     reader = csv.DictReader(infile)
    #     feature_names = reader.fieldnames
    data = pd.read_csv(os.path.join(dataset.directory, file),\
        sep='\t',\
        nrows=0) # only read 1 row for performance. we care about feature names only.
    feature_names = data.drop('Class', axis=1).columns.values
    keys = dataset['p_key'].split(',')
    p_informatives = np.where(np.isin(feature_names, keys))[0]
    p_informative = ','.join([str(i) for i in p_informatives])
    info2.at[index, 'p_informative'] = p_informative
    print(dataset.name, 'p_informatives =', p_informatives)
    # info2.iloc[index]['p_informative'] = p_informative
    # info2['p_informative'] = p_informative

info3 = info2.drop(['full_name', 'heritability','edm',\
    'p_alternate','p_key','p_predictive','replicas','description'], axis=1)
merged = all2.merge(info3, on='directory')
merged.to_csv('merged.csv', index=False)