#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : xgboost_ligh_merge1.py
#Author  : ganguohua
#Time    : 2021/10/15 8:50 下午
"""
import gc
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as  xgb
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import KFold

def params_append(params):
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'
    params['min_child_weight'] = int(params['min_child_weight'])
    params['max_depth'] = int(params['max_depth'])
    return params


def param_beyesian(train_y, train_data):
    feats = [f for f in train_data.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    train_data = xgb.DMatrix(train_data[feats], train_y)

    def xgb_cv(colsample_bytree, subsample, min_child_weight, max_depth, reg_alpha, eta, reg_lambda):
        params = {'objective': 'binary:logistic', 'early_stopping_round': 50, 'eval_metric': 'auc',
                  'colsample_bytree': max(min(colsample_bytree, 1), 0), 'subsample': max(min(subsample, 1), 0),
                  'min_child_weight': int(min_child_weight), 'max_depth': int(max_depth), 'eta': float(eta),
                  'reg_alpha': max(reg_alpha, 0), 'reg_lambda': max(reg_lambda, 0)}
        # max_depth，min_child_weight，gamman，subsample，colsample_bytree
        # 调整树的特定参数
        print(params)
        '''
        def cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None,
       metrics=(), obj=None, feval=None, maximize=None, early_stopping_rounds=None,
       fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True,
       seed=0, callbacks=None, shuffle=True):

        '''
        cv_result = xgb.cv(params, train_data, num_boost_round=1000,
                           nfold=2, seed=2, stratified=False, shuffle=False,
                           early_stopping_rounds=30, verbose_eval=False)
        return -min(cv_result['test-auc-mean'])

    xgb_bo = BayesianOptimization(
        xgb_cv, {
            'colsample_bytree': (0.5, 1),
            'subsample': (0.5, 1),
            'min_child_weight': (1, 30),
            'max_depth': (5, 12),
            'reg_alpha': (0, 5),
            'eta': (0.02, 0.2),
            'reg_lambda': (0, 5)
        }
    )
    xgb_bo.maximize()
    print(xgb_bo.max['target'], xgb_bo.max['params'])
    return xgb_bo.max['params']



def train_predict(train_data, train_y, test, params):
    feats = [f for f in train_data.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    oof_preds=np.zeros(train_data.shape[0])
    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2021, shuffle=True)
    prediction_test = np.zeros(test.shape[0])

    cv_score = []

    ESR = 30
    NBR = 10000
    VBE = 50
    test_data=xgb.DMatrix(test[feats])
    for train_part_index, eval_index in kf.split(train_data,train_y):
        train_part = xgb.DMatrix(train_data[feats].iloc[train_part_index],train_y.iloc[train_part_index])
        eval = xgb.DMatrix(train_data[feats].iloc[eval_index],train_y.iloc[eval_index])
        '''
        def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=None, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None):
        '''
        bst = xgb.train(params, train_part, NBR, [(train_part, 'train'), (eval, 'eval')],
                        verbose_eval=VBE, maximize=False, early_stopping_rounds=ESR)
        prediction_test += bst.predict(test_data)
        oof_preds[eval_index] = bst.predict(eval)
        score = np.sqrt(roc_auc_score(train_y.iloc[eval_index].values,  oof_preds[eval_index]))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    test['isDefault'] = prediction_test / 5
    test.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('./merge_xgb1.csv', index=False)
    np.savetxt("new1.csv", oof_preds, delimiter=',')
    return oof_preds,test

# 进行新的数据集的获取
my_x_train=pd.read_csv('../my_x_train.csv')
my_y_train=pd.read_csv('../my_y_train.csv')
my_test=pd.read_csv('../my_test.csv')

## 使用xgboost 的数据集进行处理
# my_params=param_beyesian(my_y_train,my_x_train)
my_params={'colsample_bytree': 0.5065780361806267, 'eta': 0.13318296074502095, 'max_depth': 5.153204249173053, 'min_child_weight': 25.00366551520674, 'reg_alpha': 3.5262429067739554, 'reg_lambda': 3.1400347657132164, 'subsample': 0.5924164568683057}
# train_data, train_y, test, params
print(my_x_train.shape)
print(my_y_train.shape)
print(my_test.shape)
oof,ans=train_predict(my_x_train,my_y_train,my_test,my_params)


