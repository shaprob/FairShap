" Importing packages "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from math import comb
from scipy.special import bernoulli
from itertools import chain, combinations
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import mod_fairShap_kfold

import numpy.matlib


import category_encoders as ce

import time

" Definig the threshold's step and the number of folds (k fold cross validation)"
step = 0.01
nFold = 5


" COMPAS Recidivism dataset "
dataset = pd.read_excel('data_compas.xlsx')
dataset = dataset[(dataset.race != 'Other') & (dataset.race != 'Hispanic') & (dataset.race != 'Native American') & (dataset.race != 'Asian')]
X = dataset.iloc[:, 0:8]
y =  dataset.loc[:,'score_risk']

from sklearn.utils import resample

Xs, ys = resample(X[y == 0], y[y == 0],
                replace=True,
                n_samples=X[y == 1].shape[0],
                random_state=123)
X = pd.concat([X[y == 1], Xs])
y = pd.concat([y[y == 1], ys])

sensitive = X['race']
#sensitive = X['sex']

split_aux = X['race'].astype(str) + y.astype(str) # Defining where to stratify when spliting data into train and test
#split_aux = X['sex'].astype(str) + y.astype(str) # Defining where to stratify when spliting data into train and test

X = X.drop('race',axis=1)
#X = X.drop('sex',axis=1)


'''
" Adult income dataset "
dataset = pd.read_csv('data_adult.csv', na_values='?').dropna()
X = dataset.iloc[:, 0:-1]
X = X.drop('fnlwgt',axis=1)
X = X.drop('education',axis=1)
#X = X.drop('relationship',axis=1)
X = X.drop('occupation',axis=1)
age1, age2, age3 = X.age<25, X.age.between(25, 60), X.age>60
X['age'] = np.select([age1, age2, age3], ['<25', '25-60', '>60'], default=None)
X['workclass'] = np.where(X['workclass'] != 'Private', 'Non-private', X['workclass'])
X['marital-status'] = np.where(X['marital-status'] == 'Married-civ-spouse', 'married', X['marital-status'])
X['marital-status'] = np.where(X['marital-status'] == 'Married-spouse-absent', 'married', X['marital-status'])
X['marital-status'] = np.where(X['marital-status'] == 'Married-AF-spouse', 'married', X['marital-status'])
X['marital-status'] = np.where(X['marital-status'] == 'Never-married', 'never-married', X['marital-status'])
X['marital-status'] = np.where(X['marital-status'] == 'Divorced', 'other', X['marital-status'])
X['marital-status'] = np.where(X['marital-status'] == 'Separated', 'other', X['marital-status'])
X['marital-status'] = np.where(X['marital-status'] == 'Widowed', 'other', X['marital-status'])
#X['race'] = np.where(X['race'] != 'White', 'Non-white', X['race'])
X['native-country'] = np.where(X['native-country'] != 'United-States', 'Non-United-States', X['native-country'])
y =  dataset.loc[:,'income']=='>50K'
y = 1*y.astype(int)

from sklearn.utils import resample

Xs, ys = resample(X[y == 0], y[y == 0],
                replace=True,
                n_samples=X[y == 1].shape[0],
                random_state=123)
X = pd.concat([X[y == 1], Xs])
y = pd.concat([y[y == 1], ys])

sensitive = X['gender']

split_aux = X['gender'].astype(str) + y.astype(str) # Defining where to stratify when spliting data into train and test

X = X.drop('gender',axis=1)
'''

'''
" LSAC dataset "
dataset = pd.read_csv('data_lsac_new.csv')
dataset[['fulltime','male','race']]=dataset[['fulltime','male','race']].astype(str)
X = dataset.iloc[:, 0:11]
X = X.drop('tier',axis=1)
y =  dataset.loc[:,'pass_bar']

from sklearn.utils import resample

Xs, ys = resample(X[y == 1], y[y == 1],
                replace=True,
                n_samples=X[y == 0].shape[0],
                random_state=123)
X = pd.concat([X[y == 0], Xs])
y = pd.concat([y[y == 0], ys])

#sensitive = X['race']
sensitive = X['male']

#split_aux = X['race'].astype(str) + y.astype(str) # Defining where to stratify when spliting data into train and test
split_aux = X['male'].astype(str) + y.astype(str) # Defining where to stratify when spliting data into train and test

#X = X.drop('race',axis=1)
X = X.drop('male',axis=1)
'''

" If stratify on all categorical variables "
#split_aux = y.astype(str)
#for ss in cols_encoder:
#    split_aux += X[ss].astype(str)

" Defining the machine learning model - Choose one of them "
model = RandomForestClassifier()

" Calculating and defining some parameters "
k_fold = KFold(nFold,shuffle=True) # Defining the k folds (with shuffle)
nAttr = X.shape[1] # Number of attributes (before one hot encoding)
nCoal = 2**nAttr # Number of all possible coalitions of attributes
transf_matrix = np.linalg.inv(mod_fairShap_kfold.tr_shap2game(nAttr)) # Transformation matrix from game to Shapley domain
thresh = np.arange(0,1+step,step) # Considered thresholds
attr_names = X.columns

" Matrices of results "
# True/false positive values and number of test samples of each group
tp, tp_sens, tp_priv = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))
tn, tn_sens, tn_priv = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))
fp, fp_sens, fp_priv = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))
fn, fn_sens, fn_priv = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))

nSens, nPriv = np.zeros((nFold,)), np.zeros((nFold,))

attr_names = X.columns

#Novel strategy
seed=50
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
rf_classifier = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')

for kk, (train, test) in enumerate(k_fold.split(X, y, groups=split_aux)):
    
    start_time = time.time()

    X_train, X_test, y_train, y_test = X.iloc[train,:], X.iloc[test,:], y.iloc[train], y.iloc[test]
    nSamp_train = X_train.shape[0] # Number of samples (train) and attributes
    nSamp_test = X_test.shape[0] # Number of samples (test)
    
    # Compas dataset
    sensitive_test = sensitive.iloc[test]=='African-American'
    #sensitive_test = sensitive.iloc[test]=='Female'
    
    # Adult income dataset
    #sensitive_test = sensitive.iloc[test]=='Female'
    
    # LSAC dataset
    #sensitive_test = sensitive.iloc[test]==0
    #sensitive_test = sensitive.iloc[test]==0
    
    #X_train = X_train.drop('race', axis=1)
    #X_test = X_test.drop('race', axis=1)
    
    nSens[kk], nPriv[kk] = np.sum(sensitive_test == True), np.sum(sensitive_test == False)
    
    for i,s in enumerate(mod_fairShap_kfold.powerset(range(nAttr),nAttr)):
        s = list(s)
        if len(s) >= 1:
            X_train_aux = X_train.iloc[:,s]
            X_test_aux = X_test.iloc[:,s]
            
            #Novel strategy
            features_to_encode = X_train_aux.columns[X_train_aux.dtypes==object].tolist()
            
            col_trans = make_column_transformer(
                                    (OneHotEncoder(),features_to_encode),
                                    remainder = "passthrough"
                                    )
            
            pipe = make_pipeline(col_trans, rf_classifier)
            pipe.fit(X_train_aux, y_train)

            " Calculating the true/false positive/negative rates "
            for jj in range(len(thresh)):
                y_pred = (pipe.predict_proba(X_test_aux)[:, 1] > thresh[jj]).astype('float')
                
                tn[i,jj,kk] = sum((y_pred == 0) * (y_test == 0))
                tp[i,jj,kk] = sum((y_pred == 1) * (y_test == 1))
                fn[i,jj,kk] = sum((y_pred == 0) * (y_test == 1))
                fp[i,jj,kk] = sum((y_pred == 1) * (y_test == 0))
                
                tn_sens[i,jj,kk] = sum((y_pred == 0) * (y_test == 0) * (sensitive_test == True))
                tp_sens[i,jj,kk] = sum((y_pred == 1) * (y_test == 1) * (sensitive_test == True))
                fn_sens[i,jj,kk] = sum((y_pred == 0) * (y_test == 1) * (sensitive_test == True))
                fp_sens[i,jj,kk] = sum((y_pred == 1) * (y_test == 0) * (sensitive_test == True))
                
                tn_priv[i,jj,kk] = sum((y_pred == 0) * (y_test == 0) * (sensitive_test == False))
                tp_priv[i,jj,kk] = sum((y_pred == 1) * (y_test == 1) * (sensitive_test == False))
                fn_priv[i,jj,kk] = sum((y_pred == 0) * (y_test == 1) * (sensitive_test == False))
                fp_priv[i,jj,kk] = sum((y_pred == 1) * (y_test == 0) * (sensitive_test == False))
    
    tp[0,:,kk] = np.round(np.flip(thresh) * np.sum(y_test==1))
    tp_sens[0,:,kk] = np.round(np.flip(thresh) * np.sum((y_test==1)*(sensitive_test == True)))
    tp_priv[0,:,kk] = tp[0,:,kk] - tp_sens[0,:,kk]
    
    tn[0,:,kk] = np.round(thresh * np.sum(y_test==0))
    tn_sens[0,:,kk] = np.round(thresh * np.sum((y_test==0)*(sensitive_test == True)))
    tn_priv[0,:,kk] = tn[0,:,kk] - tn_sens[0,:,kk]
    
    fp[0,:,kk] = np.sum(y_test==0) - tn[0,:,kk]
    fp_sens[0,:,kk] = np.round(np.flip(thresh) * np.sum((y_test==0)*(sensitive_test == True)))
    fp_priv[0,:,kk] = fp[0,:,kk] - fp_sens[0,:,kk]
    
    fn[0,:,kk] = np.sum(y_test==1) - tp[0,:,kk]
    fn_sens[0,:,kk] = np.round(thresh * np.sum((y_test==1)*(sensitive_test == True)))
    fn_priv[0,:,kk] = fn[0,:,kk] - fn_sens[0,:,kk]
 
    #print(kk)
    print("--- %s seconds ---" % (time.time() - start_time))      

tpr, fpr, ppv, npv = tp/(tp+fn), fp/(fp+tn), tp/(tp+fp), tn/(tn+fn)
tpr[np.isnan(tpr)], fpr[np.isnan(fpr)], ppv[np.isnan(ppv)], npv[np.isnan(npv)] = 0, 0, 0, 0

tpr_sens, fpr_sens, ppv_sens, npv_sens = tp_sens/(tp_sens+fn_sens), fp_sens/(fp_sens+tn_sens), tp_sens/(tp_sens+fp_sens), tn_sens/(tn_sens+fn_sens)
tpr_sens[np.isnan(tpr_sens)], fpr_sens[np.isnan(fpr_sens)], ppv_sens[np.isnan(ppv_sens)], npv_sens[np.isnan(npv_sens)] = 0, 0, 0, 0

tpr_priv, fpr_priv, ppv_priv, npv_priv = tp_priv/(tp_priv+fn_priv), fp_priv/(fp_priv+tn_priv), tp_priv/(tp_priv+fp_priv), tn_priv/(tn_priv+fn_priv)
tpr_priv[np.isnan(tpr_priv)], fpr_priv[np.isnan(fpr_priv)], ppv_priv[np.isnan(ppv_priv)], npv_priv[np.isnan(npv_priv)] = 0, 0, 0, 0

''' Finding contributions of features towards True/False positive/negative rates '''

# True/false positive/negative rates
shapley_all, shapley_sens, shapley_priv = mod_fairShap_kfold.contr_rates(tpr,tpr_sens,tpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,'TPR')
shapley_all, shapley_sens, shapley_priv = mod_fairShap_kfold.contr_rates(fpr,fpr_sens,fpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,'FPR')


# Statistical Parity
shapley_StPa_all_mean, shapley_StPa_sens_mean, shapley_StPa_priv_mean, shapley_StPa_mean, shapley_StPa_all_std, shapley_StPa_sens_std, shapley_StPa_priv_std, shapley_StPa_std = mod_fairShap_kfold.statistical_parity(tp,fp,tp_sens,fp_sens,tp_priv,fp_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Predictive Parity
shapley_PrPa_all_mean, shapley_PrPa_sens_mean, shapley_PrPa_priv_mean, shapley_PrPa_mean, shapley_PrPa_all_std, shapley_PrPa_sens_std, shapley_PrPa_priv_std, shapley_PrPa_std = mod_fairShap_kfold.predictive_parity(ppv,ppv_sens,ppv_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Predictive Equality
shapley_PrEq_all_mean, shapley_PrEq_sens_mean, shapley_PrEq_priv_mean, shapley_PrEq_mean, shapley_PrEq_all_std, shapley_PrEq_sens_std, shapley_PrEq_priv_std, shapley_PrEq_std = mod_fairShap_kfold.predictive_equality(fpr,fpr_sens,fpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Equal Opportunity
shapley_EqOp_all_mean, shapley_EqOp_sens_mean, shapley_EqOp_priv_mean, shapley_EqOp_mean, shapley_EqOp_all_std, shapley_EqOp_sens_std, shapley_EqOp_priv_std, shapley_EqOp_std = mod_fairShap_kfold.equal_opportunity(tpr,tpr_sens,tpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Equalized Odds
shapley_EqOd_mean, shapley_EqOd_std = mod_fairShap_kfold.equalized_odds(tpr,fpr,tpr_sens,fpr_sens,tpr_priv,fpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Conditional Use Accuracy Equality
shapley_CAEq_npv_mean, shapley_CAEq_npv_sens_mean, shapley_CAEq_npv_priv_mean, shapley_CAEq_mean, shapley_CAEq_npv_std, shapley_CAEq_npv_sens_std, shapley_CAEq_npv_priv_std, shapley_CAEq_std = mod_fairShap_kfold.conditional_accuracy_equality(ppv,ppv_sens,ppv_priv,npv,npv_sens,npv_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Overall Accuracy Equality
shapley_OAEq_all_mean, shapley_OAEq_sens_mean, shapley_OAEq_priv_mean, shapley_OAEq_mean, shapley_OAEq_all_std, shapley_OAEq_sens_std, shapley_OAEq_priv_std, shapley_OAEq_std = mod_fairShap_kfold.overall_accuracy_equality(tp,tn,tp_sens,tn_sens,tp_priv,tn_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Treatment equality
shapley_TrEq_all_mean, shapley_TrEq_sens_mean, shapley_TrEq_priv_mean, shapley_TrEq_mean, shapley_TrEq_all_std, shapley_TrEq_sens_std, shapley_TrEq_priv_std, shapley_TrEq_std = mod_fairShap_kfold.treatment_equality(fn,fp,fn_sens,fp_sens,fn_priv,fp_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold)

# Waterfall plot
mod_fairShap_kfold.plot_waterfall(nAttr,np.append(np.mean(fpr_sens[0,50,:]-fpr_priv[0,50,:]),shapley_PrEq_mean[1:nAttr+1,50]),np.append(np.append(0,shapley_PrEq_std[1:nAttr+1,50]),np.std(fpr_sens[-1,50,:]-fpr_priv[-1,50,:])),attr_names,'Equal Opportunity')

" Save (if it is the case) "
data_save = [tp, fp, tn, fn, tp_sens, fp_sens, tn_sens, fn_sens, tp_priv, fp_priv, tn_priv, fn_priv, nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold]
np.save('results_fairShap_compas_raceA_kfold_test.npy', data_save, allow_pickle=True)
#tp, fp, tn, fn, tp_sens, fp_sens, tn_sens, fn_sens, tp_priv, fp_priv, tn_priv, fn_priv, nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold = np.load('results_fairShap_compas_raceA_kfold_test.npy', allow_pickle=True)

