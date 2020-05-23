### Adds more info to a data collection descriptor file (e.g. `rebate.csv`),
### such as amount of features `p`, amount of samples `n`, or dataset name.

import pandas as pd

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
info3 = info2.drop(['full_name', 'heritability','edm',\
    'p_alternate','p_key','p_predictive','replicas'], axis=1)
merged = all2.merge(info3, on='directory')
merged.to_csv('merged.csv', index=False)