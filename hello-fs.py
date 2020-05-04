
##### MANUSCRIPT #####
import scipy.io as sio 
import numpy as np
from sklearn.preprocessing import scale, LabelEncoder
# label encoder
le = LabelEncoder()
# Create a class for datasets
class ft_dataset:
    def __init__(self,input,output,name,subset_size):
        self.input = input
        self.output = output
        self.name = name 
        self.subset_size = subset_size
# Pixraw10p Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_pixraw10P = sio.loadmat('Datasets/Pixraw10p/pixraw10P.mat')
X = data_pixraw10P['X']
X = scale(X)
y = data_pixraw10P['Y']
y = np.ravel(y)
y = le.fit_transform(y)
pixraw10P = ft_dataset(X,y,"Pixraw10p",np.minimum(X.shape[1],100))
print('dataset reading end')






X = pixraw10P.input
y = pixraw10P.output
k = pixraw10P.subset_size
print('dataset before FS: {}'.format(X.shape))
### CV ###
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
neigh = KNeighborsClassifier(n_neighbors=1)
print('scores before FS: {}'.format(cross_val_score(neigh, X, y)))

##### FEATURE SELECTION #####
from sklearn.feature_selection import SelectKBest
### ANOVA F-value
from sklearn.feature_selection import f_classif
X_new = SelectKBest(f_classif, k=k).fit_transform(X, y)
print('dataset after f_classif: {}'.format(X_new.shape))
print('scores after f_classif: {}'.format(cross_val_score(neigh, X_new, y)))

### Mutual Info
# from sklearn.feature_selection import mutual_info_classif
# X_new = SelectKBest(mutual_info_classif, k=k).fit_transform(X, y)
# print('dataset after mutual_info_classif: {}'.format(X_new.shape))
# print('scores after mutual_info_classif: {}'.format(cross_val_score(neigh, X_new, y)))

### MultiSURF
from skrebate import MultiSURF
from sklearn.pipeline import make_pipeline
clf = make_pipeline(MultiSURF(n_features_to_select=k), neigh)
print('scores after MultiSURF: {}'.format(cross_val_score(clf, X, y)))


#$ dataset reading end
#$ dataset before FS: (100, 10000)
#$ scores before FS: [0.85 0.9  1.   1.   1.  ]
#$ dataset after f_classif: (100, 100)
#$ scores after f_classif: [0.9  0.95 0.95 1.   1.  ]
#$ dataset after f_classif: (100, 100)
#$ scores after f_classif: [0.95 1.   0.9  1.   1.  ]