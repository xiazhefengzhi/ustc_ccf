#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : merge_example.py
#Author  : ganguohua
#Time    : 2021/10/14 9:13 上午
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  Ridge,Lasso
from math import  sqrt
from sklearn.metrics import  mean_squared_error
class SklearnWrapper(object):

    def __init__(self, clf, seed=0, params=None) -> None:
        super().__init__()
        params['random_status'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


x_train = []
y_train = []
x_test = []

kf = KFold(n_splits=5, shuffle=True, random_state=546789)
# 进行交叉验证
def get_oof(clf):
    # 构建 0 维的数据
    oof_train = np.zeros(x_train.shape[0], )
    oof_test = np.zeros(x_test.shape[0], )
    oof_test_skf = np.empty((5, x_test.shape[0]))

    for i ,(train_idex,val_idx) in enumerate(kf.split(x_train,y_train)):
        trn_x,trn_y,val_x,val_y=x_train.iloc[train_idex],\
                                y_train[train_idex],x_train.iloc[val_idx],y_train[val_idx]
        # 训练
        clf.train(trn_y,trn_y)
        # 预测
        oof_train[val_idx]=clf.predict(val_x)
        # 记录数据
        oof_test_skf[i:]=clf.predict[x_test]
    oof_test[:]=oof_test_skf.mean(axis=0)
    # 返回通过模型进行预测的数据和对test进行预测的数据
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

et_params={
    'n_estimators':100,
    'max_features':0.5,
    'max_depth':12,
    'min_samples_leaf':2
}
rf_params={
    'n_estimators':100,
    'max_features':0.2,
    'max_depth':12,
    'min_sample_leaf':2,
}
rd_params={'alpha':10}
ls_params={'alpha':0.005}
et=SklearnWrapper(clf=ExtraTreesClassifier,seed=2020,params=et_params)
rf=SklearnWrapper(clf=RandomForestClassifier,seed=2020,params=rf_params)
rd=SklearnWrapper(clf=Ridge,seed=2020,params=rd_params)
ls=SklearnWrapper(clf=Lasso,seed=2020,params=ls_params)
et_oof_train,et_oof_test=get_oof(et)
rf_oof_train,rf_oof_test=get_oof(rf)
rd_oof_train,rd_oof_test=get_oof(rd)
ls_oof_train,ls_oof_test=get_oof(ls)

def stack_model(oof1,oof2,oof3,oof4,predictions_1,predictions_2,predictions_3,predictions_4):
    train_stack=np.hstack([oof1,oof2,oof3,oof4])
    test_stack=np.hstack([predictions_1,predictions_2,predictions_3,predictions_4])
    oof=np.zeros((train_stack.shape[0]))
    predictions=np.zeros(train_stack.shape[0],)
    scores=[]
    for fold_,(trn_idx,val_idx) in enumerate(kf.split(train_stack,y_train)):
        trn_data,trn_y=train_stack[trn_idx],y_train[trn_idx]
        val_data,val_y=train_stack[val_idx],y_train[val_idx]
        clf=Ridge(random_state=2020)
        clf.fit(trn_data,trn_y)
        oof[val_idx]=clf.predict(val_data)
        predictions+=clf.predict(test_stack)/5
        scores_single=sqrt(mean_squared_error(val_y,oof[val_idx]))
        scores.append(scores_single)
        print(f'{fold_+1}/{5}',scores_single)
    print('mean',np.mean(scores))
    return oof,predictions
oof_stack,prredictions_stack=stack_model(et_oof_train,rf_oof_train,
                                         rd_oof_train,ls_oof_train,
                                         et_oof_test,rf_oof_test,rd_oof_test,ls_oof_test,
                                         y_train)

