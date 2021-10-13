#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : xgboost_search.py
#Author  : ganguohua
#Time    : 2021/10/11 8:28 上午
"""
import numpy as np
import pandas as pd

'''
“ 使用前安装：pip install bayesian-optimization
'''
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import xgboost as  xgb
from sklearn.metrics import mean_squared_error


def params_append(params):
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'rmse'
    params['min_child_weight'] = int(params['min_child_weight'])
    params['max_depth'] = int(params['max_depth'])
    return params


def param_beyesian(train_y, train_data):
    train_data = xgb.DMatrix(train_data, train_y)

    def xgb_cv(colsample_bytree, subsample, min_child_weight, max_depth, reg_alpha, eta, reg_lambda):
        params = {'objective': 'reg:squarederror', 'early_stopping_round': 50, 'eval_metric': 'rmse',
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
        return -min(cv_result['test-rmse-mean'])

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
    xgb_bo.maximize(init_points=21, n_iter=5)
    print(xgb_bo.max['target'], xgb_bo.max['params'])
    return xgb_bo.max['params']


def train_predict(train_data, train_y, test_data, params):
    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2021, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 30
    NBR = 10000
    VBE = 50
    for train_part_index, eval_index in kf.split(train_data):
        train_part = xgb.DMatrix(train_data.loc[train_part_index], train_y.loc[train_part_index])
        eval = xgb.DMatrix(train_data.loc[eval_index], train_y.loc[eval_index])
        '''
        def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=None, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None):
        '''
        bst = xgb.train(params, train_part, NBR, [(train_part, 'train'), (eval, 'eval')],
                        verbose_eval=VBE, maximize=False, early_stopping_rounds=ESR)
        prediction_test += bst.predict(test_data)
        eval_pre = bst.predict(eval)
        prediction_train = prediction_train.append(pd.Series(eval_pre, index=eval_index))
        score = np.sqrt(mean_squared_error(train_y.loc[eval_index].values, eval_pre))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    test_data['isDefault'] = prediction_test / 5
    test_data.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('./xgb1.csv', index=False)

    return
