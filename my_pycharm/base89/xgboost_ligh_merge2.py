#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : xgboost_ligh_merge2.py
#Author  : ganguohua
#Time    : 2021/10/15 9:02 下午
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedKFold

predict1 = pd.read_csv('../baseline892.csv')
predict1=np.array(predict1['isDefault'].values).reshape(-1,1)
predict2 = pd.read_csv('../nn2.csv')
predict2=np.array(predict2['isDefault'].values).reshape(-1,1)
oof1 = pd.read_csv('../new.csv', header=None, index_col=False)
oof1=np.array(oof1)
oof2 = pd.read_csv('../new.csv', header=None, index_col=False)
oof2=np.array(oof2)
y_train = pd.read_csv('../my_y_train.csv')
y_train=np.array(y_train.values)
y_train.reshape(-1)
print(len(y_train))
print(len(predict2))
print(len(oof1))

def stack_model(oof1, oof2, predictions_1, predictions_2):
    train_stack = np.hstack([oof1, oof2])
    test_stack = np.hstack([predictions_1,predictions_2])
    folds = RepeatedKFold(n_splits=4, n_repeats=2, random_state=2021)
    oof = np.zeros((train_stack.shape[0]))
    predictions = np.zeros(test_stack.shape[0], )
    scores = []

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, y_train)):
        trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
        val_data, val_y = train_stack[val_idx], y_train[val_idx]
        clf = BayesianRidge()
        clf.fit(trn_data, trn_y.reshape(-1))
        oof[val_idx] = clf.predict(val_data)
        predictions += clf.predict(test_stack) / (5 * 2)
        print('mean', log_loss(y_train, oof))
    return oof, predictions
oof, predictions = stack_model(oof1, oof2, predict1, predict2)
print(predictions)
