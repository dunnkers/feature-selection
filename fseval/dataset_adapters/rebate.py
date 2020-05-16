import pandas as pd
import os
from sklearn.utils import Bunch

def load_dataset(filename):
    dataset = pd.read_csv(filename, sep='\t')
    return Bunch(
        data=dataset.drop('Class', axis=1).values,
        target= dataset['Class'].values,
        feature_names= dataset.drop('Class', axis=1).columns.values,
        filename= os.path.abspath(filename)
    )
