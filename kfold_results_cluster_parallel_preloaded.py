# Load modules

from __future__ import division
import timeit
import os
import datetime
import scipy.io as sio 
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from boruta import BorutaPy
from skrebate import ReliefF, SURF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.datasets import make_classification
from BIFS.iterative_input_selection_classification import IISClassification as IIS
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
from tempfile import TemporaryFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import pathos.pools as pp
from functools import partial   #because nested functions and such
import random
import shutil
import math

########################################################################


# Create a class for datasets
class ft_dataset:
    def __init__(self,input,output,name,subset_size):
        self.input = input
        self.output = output
        self.name = name 
        self.subset_size = subset_size
        

########################################################################


# function for k-fold cross-validation selection. 
def kfold_selection(dataset,learners,X,y,k=5,n_features=10):
    
    X = dataset.input
    y = dataset.output 
    kf = KFold(n_splits=k, shuffle=True)  
    output1 = {}   #accuracies
    output5 = {}   #f1_scores
    output2 = {}   #output 2 is the subsets
    output3 = {}   #runtime
    output4 = {}   #correlations
    k_cnt = 0
    n_classes = len(set(y))

    # dimensions
    n_samples = X.shape[0]
    n_inputs = X.shape[1]
    n_points = n_samples*n_inputs

    # change the number of selected features
    if n_inputs<=100:
        n_select = n_inputs
    else:
        n_select = 100


    # initialize the output for each learner
    # output 1 is accuracies, output 2 is the subsets, output3 is runtime
    for model in learners:
        output1.update({type(model).__name__ : np.full((5,n_features,k),np.nan,dtype=float)})
        output5.update({type(model).__name__ : np.full((5,n_features,k),np.nan,dtype=float)})
        output4.update({type(model).__name__ : np.full((5,n_features,k),np.nan,dtype=float)})


    # loop over k-folds
    for fs_index, test_index in kf.split(X):
        # load subsets
        preload_subsets = np.load("%s/results_subsets%s_%02d.npy" % (preload_folder,dataset.name,k_cnt),allow_pickle=True)
        preload_times = np.load("%s/results_times%s_%02d.npy" % (preload_folder,dataset.name,k_cnt),allow_pickle=True)

        #print "fold %02d has started" % (k_cnt+1)
        X_fs, X_test = X[fs_index], X[test_index]
        y_fs, y_test = y[fs_index], y[test_index]

        # get subsets form the train data (XGBoost)
        print "XGB subset selection started"
        start = timeit.default_timer()

        # NEW: col_sample for large datasets
        XGB = XGBClassifier(n_estimators=100,seed=275,max_depth=3) 
        XGB_deep = XGBClassifier(n_estimators=100,seed=275,max_depth=20) 
        knn = KNeighborsClassifier(n_neighbors=1) 
        et = ExtraTreesClassifier(n_estimators=100)

        XGB_deep.fit(X_fs,y_fs)
        #rank_xgb = XGB.feature_importances_
        fscore = XGB_deep.get_booster().get_score(importance_type='gain')
        feature_importance = np.zeros(X.shape[1])
        for o, v in fscore.iteritems():
            feature_importance[int(o[1:])] = v
        rank_xgb = np.argsort(feature_importance)
        subset_xgb = rank_xgb[::-1]

        #subset_xgb = np.argsort(-rank_xgb)
        temp = [i for i in rank_xgb if i>np.mean(rank_xgb)]
        # without limiting
        subset_xgb_full = subset_xgb[:n_features]
        # with limiting
        subset_xgb = subset_xgb[:len(temp)]   #subset_xgb[:n_features]
        
        
        end = timeit.default_timer()
        output3.update({"subset_xgb%02d" % (k_cnt) : end-start})
        output2.update({"subset_xgb%02d" % (k_cnt) : subset_xgb})
        output2.update({"subset_xgb_full%02d" % (k_cnt) : subset_xgb_full})
        print "Fold %02d XGB subset:" % (k_cnt+1)
        print subset_xgb
        
        # IIS1
        print "IIS1 subset selection started"
        start = timeit.default_timer()
        IIS_C1 = IIS(estimator = XGB_deep , number_of_folds =3, epsilon=1e-18,
        siso_ranking_size = 5,siso_order=1,max_number_of_features = n_select,metric='acc',learning_rate=1.5,slow_mode=True,verbose=0)
        IIS_C1.fit(X = X_fs, Y = np.ravel(y_fs))
        subset_iis1 = IIS_C1.selected_subset_


        end = timeit.default_timer()
        stopping = IIS_C1.stopping_condition_
        output3.update({"subset_iis1%02d" % (k_cnt) : end-start})
        output3.update({"subset_iis1%02dstoping" % (k_cnt) : stopping})
        output2.update({"subset_iis1%02d" % (k_cnt) : subset_iis1})
        print "Fold %02d IIS1 subset:" % (k_cnt+1)
        print subset_iis1
        

        # IIS2
        print "IIS2 subset selection started"
        start = timeit.default_timer()
        IIS_C2 = IIS(estimator = XGB_deep,number_of_folds =3, epsilon=1e-18,
        siso_ranking_size = 200,siso_order=1,max_number_of_features = n_select,metric='acc',learning_rate=1.5,slow_mode=True,verbose=0)
        
        IIS_C2.fit(X = X_fs, Y = np.ravel(y_fs))
        subset_iis2 = IIS_C2.selected_subset_


        end = timeit.default_timer()
        stopping = IIS_C2.stopping_condition_
        output3.update({"subset_iis2%02d" % (k_cnt) : end-start})
        output3.update({"subset_iis2%02dstoping" % (k_cnt) : stopping})
        output2.update({"subset_iis2%02d" % (k_cnt) : subset_iis2})
        print "Fold %02d IIS2 subset:" % (k_cnt+1)
        print subset_iis2
        
            
        # ReliefF
        print "ReliefF subset selection started"
        """
        start = timeit.default_timer()
        Relief = ReliefF(n_features_to_select = n_select,n_neighbors=10)
        Relief.fit(X_fs,y_fs)
        Relief_score = Relief.feature_importances_

        temp = [i for i in Relief_score if i>np.mean(Relief_score)]
        # full
        subset_Relief_full = np.argsort(-Relief_score)[:n_features]
        # limited
        subset_Relief = np.argsort(-Relief_score)[:len(temp)]
        end = timeit.default_timer()
        """
        preload_time_relief = preload_times.item().get("subset_relief%02d" % k_cnt)
        subset_Relief = preload_subsets.item().get("subset_relief%02d" % k_cnt)
        subset_Relief_full = preload_subsets.item().get("subset_relief_full%02d" % k_cnt)
        print subset_Relief_full

        output3.update({"subset_relief%02d" % (k_cnt) : preload_time_relief})
        output2.update({"subset_relief%02d" % (k_cnt) : subset_Relief})
        output2.update({"subset_relief_full%02d" % (k_cnt) : subset_Relief_full})
        print "Fold %02d Relief subset:" % (k_cnt+1)
        print subset_Relief

        # Boruta
        print "Boruta subset selection started"
        """
        start = timeit.default_timer()
        #GBM = GradientBoostingClassifier(n_estimators=100)
        boruta = BorutaPy(estimator = XGB, verbose=0, random_state=275)
        boruta.fit(X_fs,y_fs)
        boruta_rank = boruta.ranking_
        boruta_count = boruta.n_features_
        subset_boruta_unsorted = boruta_rank.argsort()[:boruta_count]

        # rank the relevant selected by Boruta with the estimator (if not empty)
        if boruta_count == 0:
            subset_boruta = subset_boruta_unsorted
        else:
            XGB.fit(X_fs[:,subset_boruta_unsorted],y_fs)

            fscore = XGB.get_booster().get_score(importance_type='gain')
            feature_importance = np.zeros(len(subset_boruta_unsorted))
            for o, v in fscore.iteritems():
                feature_importance[int(o[1:])] = v
            subset_boruta = subset_boruta_unsorted[np.argsort(-feature_importance)]

        end = timeit.default_timer()
        """
        preload_time_boruta = preload_times.item().get("subset_boruta%02d" % k_cnt)
        subset_boruta = preload_subsets.item().get("subset_boruta%02d" % k_cnt)

        output3.update({"subset_boruta%02d" % (k_cnt) : preload_time_boruta})
        output2.update({"subset_boruta%02d" % (k_cnt) : subset_boruta })
        print "Fold %02d Boruta subset:" % (k_cnt+1)
        print subset_boruta

        # note: train on a subset of the train_set from which the subsets were found
        #train_index = np.random.choice(fs_index,np.round(0.8*len(fs_index)).astype(int))
        train_index = fs_index
        X_train, y_train = X[train_index], y[train_index]


        for model in learners:
            all_accuracies = np.full((5,n_features,k),np.nan,dtype=float)

            for i in range(n_features):

                # Find average pair-wise correlations
                n_cor = i+1
                xgb_cov = X_test[:,subset_xgb[0:i+1]]
                iis1_cov = X_test[:,subset_iis1[0:i+1]]
                iis2_cov = X_test[:,subset_iis2[0:i+1]]
                relief_cov = X_test[:,subset_Relief[0:i+1]]
                boruta_cov = X_test[:,subset_boruta[0:i+1]]
                

                #print "selected feature %02d is being fit" % (i+1)
                
                if i<len(subset_xgb):
                    model.fit(X_train[:,subset_xgb[0:i+1]],y_train)
                    y_test_hat_xgb = model.predict(X_test[:,subset_xgb[0:i+1]])
                    temp = output1[type(model).__name__]
                    temp[0,i,k_cnt] = accuracy_score(y_test,y_test_hat_xgb)
                    output1.update({type(model).__name__ : temp})

                    tempf = output5[type(model).__name__]
                    if n_classes==2:
                        tempf[0,i,k_cnt] = f1_score(y_test,y_test_hat_xgb)
                    else:
                        tempf[0,i,k_cnt] = f1_score(y_test,y_test_hat_xgb,average='micro')
                    output5.update({type(model).__name__ : tempf})

                    # new 
                    xgb_cov = np.cov(X_test[:,subset_xgb[0:i+1]])
                    temp_cor = output4[type(model).__name__]
                    temp_cor[0,i,k_cnt] = ((np.sum(np.sum(xgb_cov))- n_cor) / 2) / n_cor
                    output4.update({type(model).__name__ : temp_cor})

                if i<len(subset_iis1):
                    model.fit(X_train[:,subset_iis1[0:i+1]],y_train)
                    y_test_hat_iis1 = model.predict(X_test[:,subset_iis1[0:i+1]])
                    temp = output1[type(model).__name__]
                    temp[1,i,k_cnt] = accuracy_score(y_test,y_test_hat_iis1)
                    output1.update({type(model).__name__ : temp})

                    tempf = output5[type(model).__name__]
                    if n_classes==2:
                        tempf[1,i,k_cnt] = f1_score(y_test,y_test_hat_iis1)
                    else:
                        tempf[1,i,k_cnt] = f1_score(y_test,y_test_hat_iis1,average='micro')
                    output5.update({type(model).__name__ : tempf})

                    # new 
                    iis1_cov = np.cov(X_test[:,subset_iis1[0:i+1]])
                    temp_cor = output4[type(model).__name__]
                    temp_cor[1,i,k_cnt] = ((np.sum(np.sum(iis1_cov))- n_cor) / 2) / n_cor
                    output4.update({type(model).__name__ : temp_cor})
                
                
                if i<len(subset_iis2):
                    model.fit(X_train[:,subset_iis2[0:i+1]],y_train)
                    y_test_hat_iis2 = model.predict(X_test[:,subset_iis2[0:i+1]]) 
                    temp = output1[type(model).__name__]
                    temp[2,i,k_cnt] = accuracy_score(y_test,y_test_hat_iis2)
                    output1.update({type(model).__name__ : temp})

                    tempf = output5[type(model).__name__]
                    if n_classes==2:
                        tempf[2,i,k_cnt] = f1_score(y_test,y_test_hat_iis2)
                    else:
                        tempf[2,i,k_cnt] = f1_score(y_test,y_test_hat_iis2,average='micro')
                    output5.update({type(model).__name__ : tempf})

                    # new 
                    iis2_cov = np.cov(X_test[:,subset_iis2[0:i+1]])
                    temp_cor = output4[type(model).__name__]
                    temp_cor[2,i,k_cnt] = ((np.sum(np.sum(iis2_cov))- n_cor) / 2) / n_cor
                    output4.update({type(model).__name__ : temp_cor})
                
                if i<len(subset_Relief):
                    model.fit(X_train[:,subset_Relief[0:i+1]],y_train)
                    y_test_hat_relief = model.predict(X_test[:,subset_Relief[0:i+1]])
                    temp = output1[type(model).__name__]
                    temp[3,i,k_cnt] = accuracy_score(y_test,y_test_hat_relief)
                    output1.update({type(model).__name__ : temp})

                    tempf = output5[type(model).__name__]
                    if n_classes==2:
                        tempf[3,i,k_cnt] = f1_score(y_test,y_test_hat_relief)
                    else:
                        tempf[3,i,k_cnt] = f1_score(y_test,y_test_hat_relief,average='micro')
                    output5.update({type(model).__name__ : tempf})

                    # new 
                    relief_cov = np.cov(X_test[:,subset_Relief[0:i+1]])
                    temp_cor = output4[type(model).__name__]
                    temp_cor[3,i,k_cnt] = ((np.sum(np.sum(relief_cov))- n_cor) / 2) / n_cor
                    output4.update({type(model).__name__ : temp_cor})

                if i<len(subset_boruta):
                    model.fit(X_train[:,subset_boruta[0:i+1]],y_train)
                    y_test_hat_boruta = model.predict(X_test[:,subset_boruta[0:i+1]]) 
                    temp = output1[type(model).__name__]
                    temp[4,i,k_cnt] = accuracy_score(y_test,y_test_hat_boruta)
                    output1.update({type(model).__name__ : temp})

                    tempf = output5[type(model).__name__]
                    if n_classes==2:
                        tempf[4,i,k_cnt] = f1_score(y_test,y_test_hat_boruta)
                    else:
                        tempf[4,i,k_cnt] = f1_score(y_test,y_test_hat_boruta,average='micro')
                    output5.update({type(model).__name__ : tempf})

                    # new 
                    boruta_cov = np.cov(X_test[:,subset_boruta[0:i+1]])
                    temp_cor = output4[type(model).__name__]
                    temp_cor[4,i,k_cnt] = ((np.sum(np.sum(boruta_cov))- n_cor) / 2) / n_cor
                    output4.update({type(model).__name__ : temp_cor})

        k_cnt+=1      
    return output1,output2,output3,output4,output5




########################################################################
# Loading datasets

# Datasets
le = LabelEncoder()
data_sonar = np.loadtxt(open("Datasets/Sonar_Input", "rb"), delimiter=",")
X = data_sonar[:,0:-1]
X = scale(X)
y = data_sonar[:,-1]
y = le.fit_transform(y)
sonar = ft_dataset(X,y,"Sonar",np.minimum(X.shape[1],100))

data_QSAR = np.loadtxt(open("Datasets/QSAR_Input", "rb"), delimiter=",")
X = data_QSAR[:,0:-1]
X = scale(X)
y = data_QSAR[:,-1]
y = le.fit_transform(y)
qsar = ft_dataset(X,y,"QSAR",np.minimum(X.shape[1],100))

data_spam = np.loadtxt(open("Datasets/Spam_Input", "rb"), delimiter=",")
X = data_spam[:,0:-1]
X = scale(X)
y = data_spam[:,-1]
y = le.fit_transform(y)
spam = ft_dataset(X,y,"Spam",np.minimum(X.shape[1],100))

data_ionsphere = np.loadtxt(open("Datasets/Ionosphere_Input", "rb"), delimiter=",")
X = data_ionsphere[:,0:-1]
X = scale(X)
y = data_ionsphere[:,-1]
y = le.fit_transform(y)
ionsphere = ft_dataset(X,y,"Ionsphere",np.minimum(X.shape[1],100))

data_basehock = sio.loadmat('Datasets/BASEHOCK.mat')
X = data_basehock['X']
X = scale(X)
y = data_basehock['Y']
y = np.ravel(y)
y = le.fit_transform(y)
basehock = ft_dataset(X,y,'Basehock',np.minimum(X.shape[1],100))

data_pcmac = sio.loadmat('Datasets/PCMAC.mat')
X = data_pcmac['X']
X = scale(X)
y = data_pcmac['Y']
y = np.ravel(y)
y = le.fit_transform(y)
pcmac = ft_dataset(X,y,'PCMAC',np.minimum(X.shape[1],100))

data_relathe = sio.loadmat('Datasets/RELATHE.mat')
X = data_relathe['X']
X = scale(X)
y = data_relathe['Y']
y = np.ravel(y)
y = le.fit_transform(y)
relathe = ft_dataset(X,y,'RELATHE',np.minimum(X.shape[1],100))

data_madelon = sio.loadmat('Datasets/madelon.mat')
X = data_madelon['X']
X = scale(X)
y = data_madelon['Y']
y = np.ravel(y)
y = le.fit_transform(y)
madelon = ft_dataset(X,y,'Madelon',np.minimum(X.shape[1],100))


# Authorship Dataset
# URL = https://www.openml.org/d/970
data_authorship = np.loadtxt(open("Datasets/Authorship/authorship.txt", "rb"), delimiter=",")
X = data_authorship[:,0:-1]
X = scale(X)
y = data_authorship[:,-1]
y = le.fit_transform(y)
authorship = ft_dataset(X,y,"Authorship",np.minimum(X.shape[1],100))

# Hill Valley Dataset
# URL = https://www.openml.org/d/1479
data_hill_valley = np.loadtxt(open("Datasets/Hill_Valley/hill_valley.txt", "rb"), delimiter=",")
X = data_hill_valley[:,0:-1]
X = scale(X)
y = data_hill_valley[:,-1]
y = le.fit_transform(y)
hill_valley = ft_dataset(X,y,"Hill_Valley",np.minimum(X.shape[1],100))

# Oil Spill Dataset
# URL = https://www.openml.org/d/311
data_oil_spill = np.loadtxt(open("Datasets/Oil_Spill/Oil_Spill.txt", "rb"), delimiter=",")
X = data_oil_spill[:,1:]
X = scale(X)
y = data_oil_spill[:,0]
y = le.fit_transform(y)
oil_spill = ft_dataset(X,y,"Oil_Spill",np.minimum(X.shape[1],100))

# Opt Digits Dataset
# URL = https://www.openml.org/d/980
data_opt_digits = np.loadtxt(open("Datasets/Opt_Digits/optdigits.txt", "rb"), delimiter=",")
X = data_opt_digits[:,0:-1]
X = scale(X)
y = data_opt_digits[:,-1]
y = le.fit_transform(y)
opt_digits = ft_dataset(X,y,"Opt_Digits",np.minimum(X.shape[1],100))

# Ozone Level Dataset
# URL = https://www.openml.org/d/1487
data_ozone_level = np.loadtxt(open("Datasets/Ozone_Level/ozone_level.txt", "rb"), delimiter=",")
X = data_ozone_level[:,0:-1]
X = scale(X)
y = data_ozone_level[:,-1]
y = le.fit_transform(y)
ozone_level = ft_dataset(X,y,"Ozone_Level",np.minimum(X.shape[1],100))

# Parkinsons Dataset
# URL = https://www.openml.org/d/1488
data_parkinsons = np.loadtxt(open("Datasets/Parkinsons/Parkinsons.txt", "rb"), delimiter=",")
X = data_parkinsons[:,0:-1]
X = scale(X)
y = data_parkinsons[:,-1]
y = le.fit_transform(y)
parkinsons = ft_dataset(X,y,"Parkinsons",np.minimum(X.shape[1],100))


# Gisette Dataset
# URL = http://featureselection.asu.edu/datasets.php
data_gisette = sio.loadmat('Datasets/Gisette/Gisette.mat')
X = data_gisette['X']
X = scale(X)
y = data_gisette['Y']
y = le.fit_transform(y)
gisette = ft_dataset(X,y,"Gisette",np.minimum(X.shape[1],100))


# COIL20 Dataset
# URL = http://featureselection.asu.edu/datasets.php
data_coil20 = sio.loadmat('Datasets/COIL20/COIL20.mat')
X = data_coil20['X']
X = scale(X)
y = data_coil20['Y']
y = np.ravel(y)
y = le.fit_transform(y)
coil20 = ft_dataset(X,y,"Coil20",np.minimum(X.shape[1],100))

# Colon Dataset
# URL = http://featureselection.asu.edu/datasets.php
data_colon = sio.loadmat('Datasets/Colon/colon.mat')
X = data_colon['X']
X = scale(X)
y = data_colon['Y']
y = np.ravel(y)
y = le.fit_transform(y)
colon = ft_dataset(X,y,"Colon",np.minimum(X.shape[1],100))


# Isolet Dataset
# URL = http://featureselection.asu.edu/datasets.php
data_isolet = sio.loadmat('Datasets/Isolet/Isolet.mat')
X = data_isolet['X']
X = scale(X)
y = data_isolet['Y']
y = np.ravel(y)
y = le.fit_transform(y)
isolet = ft_dataset(X,y,"Isolet",np.minimum(X.shape[1],100))

# CLL-SUB-111 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_cll_sub = sio.loadmat('Datasets/CLL-SUB-111/CLL-SUB-111.mat')
X = data_cll_sub['X']
X = scale(X)
y = data_cll_sub['Y']
y = np.ravel(y)
y = le.fit_transform(y)
cll_sub = ft_dataset(X,y,"CLL-SUB-111",np.minimum(X.shape[1],100))

# Pixraw10p Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_pixraw10P = sio.loadmat('Datasets/Pixraw10p/pixraw10P.mat')
X = data_pixraw10P['X']
X = scale(X)
y = data_pixraw10P['Y']
y = np.ravel(y)
y = le.fit_transform(y)
pixraw10P = ft_dataset(X,y,"Pixraw10p",np.minimum(X.shape[1],100))

# Orlraws10p Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_orlraws10P = sio.loadmat('Datasets/Orlraws10p/orlraws10P.mat')
X = data_orlraws10P['X']
X = scale(X)
y = data_orlraws10P['Y']
y = np.ravel(y)
y = le.fit_transform(y)
orlraws10P = ft_dataset(X,y,"Orlraws10p",np.minimum(X.shape[1],100))

# SMK-CAN-187 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_smk_can = sio.loadmat('Datasets/SMK-CAN-187/SMK-CAN-187.mat')
X = data_smk_can['X']
X = scale(X)
y = data_smk_can['Y']
y = np.ravel(y)
y = le.fit_transform(y)
smk_can = ft_dataset(X,y,"SMK-CAN-187",np.minimum(X.shape[1],100))

# TOX-171 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_tox = sio.loadmat('Datasets/TOX-171/TOX-171.mat')
X = data_tox['X']
X = scale(X)
y = data_tox['Y']
y = np.ravel(y)
y = le.fit_transform(y)
tox = ft_dataset(X,y,"TOX-171",np.minimum(X.shape[1],100))

# GLI-85 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_gli = sio.loadmat('Datasets/GLI-85/GLI-85.mat')
X = data_gli['X']
X = scale(X)
y = data_gli['Y']
y = np.ravel(y)
y = le.fit_transform(y)
gli = ft_dataset(X,y,"GLI-85",np.minimum(X.shape[1],100))

# GLA-BRA-180 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_gla_bra = sio.loadmat('Datasets/GLA-BRA-180/GLA-BRA-180.mat')
X = data_gla_bra['X']
X = scale(X)
y = data_gla_bra['Y']
y = np.ravel(y)
y = le.fit_transform(y)
gla_bra = ft_dataset(X,y,"GLA-BRA-180",np.minimum(X.shape[1],100))

# warpPIE10p Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_warpPIE10p = sio.loadmat('Datasets/warpPIE10p/warpPIE10P.mat')
X = data_warpPIE10p['X']
X = scale(X)
y = data_warpPIE10p['Y']
y = np.ravel(y)
y = le.fit_transform(y)
warpPIE10p = ft_dataset(X,y,"warpPIE10p",np.minimum(X.shape[1],100))

# warpAR10p Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_warpAR10p = sio.loadmat('Datasets/warpAR10p/warpAR10P.mat')
X = data_warpAR10p['X']
X = scale(X)
y = data_warpAR10p['Y']
y = np.ravel(y)
y = le.fit_transform(y)
warpAR10p = ft_dataset(X,y,"warpAR10p",np.minimum(X.shape[1],100))

# Yale_32x32 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_yale_32x32 = sio.loadmat('Datasets/Yale_32x32/Yale_32x32.mat')
X = data_yale_32x32['fea']
X = scale(X)
y = data_yale_32x32['gnd']
y = np.ravel(y)
y = le.fit_transform(y)
yale_32x32 = ft_dataset(X,y,"Yale_32x32",np.minimum(X.shape[1],100))

# Yale_64x64 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_yale_64x64 = sio.loadmat('Datasets/Yale_64x64/Yale_64x64.mat')
X = data_yale_64x64['fea']
X = scale(X)
y = data_yale_64x64['gnd']
y = np.ravel(y)
y = le.fit_transform(y)
yale_64x64 = ft_dataset(X,y,"Yale_64x64",np.minimum(X.shape[1],100))

# ORL_32x32 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_orl_32x32 = sio.loadmat('Datasets/ORL_32x32/ORL_32x32.mat')
X = data_orl_32x32['fea']
X = scale(X)
y = data_orl_32x32['gnd']
y = np.ravel(y)
y = le.fit_transform(y)
orl_32x32 = ft_dataset(X,y,"ORL_32x32",np.minimum(X.shape[1],100))

# ORL_64x64 Dataset
# URL = http://featureselection.asu.edu/old/datasets.php
data_orl_64x64 = sio.loadmat('Datasets/ORL_64x64/ORL_64x64.mat')
X = data_orl_64x64['fea']
X = scale(X)
y = data_orl_64x64['gnd']
y = np.ravel(y)
y = le.fit_transform(y)
orl_64x64 = ft_dataset(X,y,"ORL_64x64",np.minimum(X.shape[1],100))


########################################################################

# Experiments 
datasets = [tox,madelon,basehock,pcmac,relathe,coil20,orl_32x32,
            orlraws10P,pixraw10P,warpAR10p,warpPIE10p,gli,gla_bra,smk_can,cll_sub,
            colon,isolet,gisette]


preload_folder = "kfold_results_2019-07-31_231945CompleteKnnAndXgb"


FT_names = ["XGB","IIS_Deep_5","IIS_Deep_200","ReliefF","BORUTA"]


time_started = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
os.mkdir("kfold_results_%s" % time_started)

shutil.copy(__file__, "kfold_results_%s/settings.py" % time_started) 


def parallel_fun(m):

    for ds in datasets: 
        np.random.seed(m+99)
        learners = [KNeighborsClassifier(n_neighbors=1),DecisionTreeClassifier(max_depth=5,random_state=275),GaussianNB(),XGBClassifier(n_estimators=100,seed=275)] 
        learner_names = ["KNN","TREE","Guassian NB","XGB"]
        dataset_name = ds.name
        subset_size = ds.subset_size
        print "%s job %02d has started" % (dataset_name, m) 
        X = ds.input
        y = ds.output
        fig=plt.figure(figsize=(10, 10), dpi= 120, facecolor='w', edgecolor='k')
        cnt = 0
        all_accuracies,all_subsets,all_times,all_correlations,all_f1scores = kfold_selection(ds,learners,X,y,k=3,n_features=subset_size)
        
        for model in learners: 
            current_accuracies = all_accuracies[type(model).__name__]    
            ax = plt.subplot(2,2,cnt+1)
            ax.plot(range(subset_size)+np.ones(subset_size), np.nanmean(current_accuracies[0,:,:],axis=1),'go--')
            ax.plot(range(subset_size)+np.ones(subset_size), np.nanmean(current_accuracies[1,:,:],axis=1),'b^--')
            ax.plot(range(subset_size)+np.ones(subset_size), np.nanmean(current_accuracies[2,:,:],axis=1),'c+--')
            ax.plot(range(subset_size)+np.ones(subset_size), np.nanmean(current_accuracies[3,:,:],axis=1),'rx--')
            ax.plot(range(subset_size)+np.ones(subset_size), np.nanmean(current_accuracies[4,:,:],axis=1),'ys--')
            ax.grid(True)
            ax.set_ylim([0,1])
            title_str = dataset_name + "_" + type(model).__name__
            plt.title(title_str)
            plt.xlabel("Number of features selected")
            plt.ylabel("Average Classification accuracy")
            plt.legend((FT_names[0], FT_names[1], FT_names[2],FT_names[3],FT_names[4]),loc='best')
            cnt+=1

        print "%s job %02d has ended" % (dataset_name, m)
        if m<3:
            fig.savefig("kfold_results_%s/results_%s_%02d.pdf" % (time_started,dataset_name,m),bbox_inches="tight") 
        np.save("kfold_results_%s/results_accuracies%s_%02d" % (time_started,dataset_name,m),all_accuracies)
        np.save("kfold_results_%s/results_subsets%s_%02d" % (time_started,dataset_name,m),all_subsets)
        np.save("kfold_results_%s/results_times%s_%02d" % (time_started,dataset_name,m),all_times)
        np.save("kfold_results_%s/results_correlations%s_%02d" % (time_started,dataset_name,m),all_correlations)
        np.save("kfold_results_%s/results_f1scores%s_%02d" % (time_started,dataset_name,m),all_f1scores)


if __name__ == '__main__':
    poolSize = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) # Amount of CPUs requested.
    repeats = list(range(poolSize))
    pool = multiprocessing.Pool(processes=poolSize,) 
    pool.map(parallel_fun, repeats)
    pool.close() # Stop pool accordingly.
    pool.join()  # Wrap up data from the workers in the pool.

