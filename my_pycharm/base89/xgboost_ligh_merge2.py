#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : xgboost_ligh_merge2.py
#Author  : ganguohua
#Time    : 2021/10/15 9:02 下午
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  Ridge,Lasso
from sklearn.linear_model import  BayesianRidge
from math import  sqrt
from sklearn.metrics import  mean_squared_error,log_loss
from sklearn.model_selection import  RepeatedKFold
predict1 = pd.read_csv('../baseline892.csv')
predict2 = pd.read_csv('../nn2.csv')
oof1=np.array(pd.read_csv('../new.csv'))
oof2=np.array(pd.read_csv('../new.csv'))
y_train=pd.read_csv('../my_y_train.csv')
print(y_train.head())
def stack_model(oof1,oof2,predictions_1,predictions_2):
    train_stack=np.hstack([oof1,oof2])
    test_stack=np.hstack([predictions_1,predictions_2])
    folds=RepeatedKFold(n_splits=5,n_repeats=2,random_state=2021)
    oof=np.zeros((train_stack.shape[0]))
    predictions=np.zeros(train_stack.shape[0],)
    scores=[]

    for fold_,(trn_idx,val_idx) in enumerate(folds.split(train_stack,y_train)):
        trn_data,trn_y=train_stack[trn_idx],y_train[trn_idx]
        val_data,val_y=train_stack[val_idx],y_train[val_idx]
        clf=BayesianRidge(random_state=2020)
        clf.fit(trn_data,trn_y)
        oof[val_idx]=clf.predict(val_data)
        predictions+=clf.predict(test_stack)/(5*2)
        print('mean',log_loss(y_train,oof))

    return oof,predictions

oof,predictions=stack_model(oof1,oof2,predict1,predict2)
print(predictions)

