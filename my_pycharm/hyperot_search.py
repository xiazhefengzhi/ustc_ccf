#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : hyperot_search.py
#Author  : ganguohua
#Time    : 2021/10/9 2:08 下午
"""
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import Trials, fmin, STATUS_OK, tpe
from sklearn import  datasets
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import cross_val_score
sns.set(style="whitegrid", palette="husl")
import pandas as pd
iris=datasets.load_iris()
X=iris.data
y=iris.target
def hyperopt_train_test(params):
    clf=KNeighborsClassifier(**params)
    return cross_val_score(clf,X,y).mean()
space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,100))
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print (best)
for i in trials:
    print(i)

