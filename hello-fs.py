
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

# Colon Dataset
# URL = http://featureselection.asu.edu/datasets.php
data_colon = sio.loadmat('Datasets/Colon/colon.mat')
X = data_colon['X']
X = scale(X)
y = data_colon['Y']
y = np.ravel(y)
y = le.fit_transform(y)
colon = ft_dataset(X,y,"Colon",np.minimum(X.shape[1],100))


## Select dataset.
X = colon.input
y = colon.output
k = colon.subset_size
print('dataset reading end')


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
# from skrebate import MultiSURF
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(MultiSURF(n_features_to_select=k), neigh)
# print('scores after MultiSURF: {}'.format(cross_val_score(clf, X, y)))

### FeatBoost
from feat_boost import FeatBoostClassification
from xgboost import XGBClassifier
xgboost_ensemble = XGBClassifier(max_depth=3, learning_rate=0.1,\
    n_estimators=200, silent=True, objective='binary:logistic',\
    booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,\
    max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,\
    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,\
    random_state=0, seed=None, missing=None)
featboost = FeatBoostClassification(estimator=[xgboost_ensemble,\
    xgboost_ensemble, xgboost_ensemble], number_of_folds = 10,\
        siso_ranking_size = 8,\
        max_number_of_features = 100,\
        siso_order=4,\
        epsilon=1e-18,\
        verbose=2)
x = featboost.fit(X, y)
print('FeatBoost accuracies: {}'.format(featboost.accuracy_))
X_new = list(map(lambda x: np.take(x, featboost.selected_subset_), X))
print('scores after featboost: {}'.format(cross_val_score(neigh, X_new, y)))