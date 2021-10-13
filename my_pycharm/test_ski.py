#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : test_ski.py
#Author  : ganguohua
#Time    : 2021/10/10 3:56 下午
"""
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

class FixedBayesSearchCV(BayesSearchCV):
    """
    Dirty hack to avoid compatibility issues with sklearn 0.2 and skopt.
    Credit: https://www.kaggle.com/c/home-credit-default-risk/discussion/64004

    For context, on why the workaround see:
        - https://github.com/scikit-optimize/scikit-optimize/issues/718
        - https://github.com/scikit-optimize/scikit-optimize/issues/762
    """
    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                n_points=1, iid=True, refit=True, cv=None, verbose=0,
                pre_dispatch='2*n_jobs', random_state=None,
                error_score='raise', return_train_score=False):
        """
        See: https://github.com/scikit-optimize/scikit-optimize/issues/762#issuecomment-493689266
        """

        # Bug fix: Added this line
        self.fit_params = fit_params

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)

        # Removed the passing of fit_params to the parent class.
        super(BayesSearchCV, self).__init__(
                estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
                refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
                error_score=error_score, return_train_score=return_train_score)

    def _run_search(self, x):
        raise BaseException('Use newer skopt')

model_lgb = lgb.LGBMClassifier(
            learning_rate=0.1,   # 学习率
            n_estimators=10000,    # 树的个数
            max_depth=10,         # 树的最大深度
            num_leaves=31,        # 叶子节点个数 'leaf-wise'
            min_split_gain=0,     # 节点分裂所需的最小损失函数降低值
            objective='multiclass', # 多分类
            metric='multiclass',  # 评价函数
            num_class=4,          # 多分类问题类别数
            subsample=0.8,        # 样本随机采样做为训练集的比例
            colsample_bytree=0.8, # 使用特征比例
            seed=1)

# 若包含类别变量，将其类型设置为category，astype('category')
# lightgbm scikit-optimize
def lgb_auto_para_tuning_bayesian(model_lgb,X,Y):
    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.80, random_state=0)
    # cv：交叉验证 n_points：并行采样的超参组数
    opt = FixedBayesSearchCV(model_lgb,cv=3,n_points=2,n_jobs=4,verbose=1,
        search_spaces={
            'learning_rate': Real(0.008, 0.01),
            'max_depth': Integer(3, 10),
            'num_leaves': Integer(31, 127),
            'min_split_gain':Real(0.0,0.4),
            'min_child_weight':Real(0.001,0.002),
            'min_child_samples':Integer(18,22),
            'subsample':Real(0.6,1.0),
            'subsample_freq':Integer(3,5),
            'colsample_bytree':Real(0.6,1.0),
            'reg_alpha':Real(0,0.5),
            'reg_lambda':Real(0,0.5)
        },
         fit_params={
                 'eval_set':[(test_x, test_y)],
                 'eval_metric': 'multiclass',
                 'early_stopping_rounds': 50
                 })
    opt.fit(train_x,train_y)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(test_x, test_y))
    print("Best parameters: ", opt.best_params_)
    print("Best estimator:", opt.best_estimator_)

model_xgb = XGBClassifier(
            learning_rate =0.01,  # 学习率
            n_estimators=10000,   # 树的个数
            max_depth=6,         # 树的最大深度
            min_child_weight=1,  # 叶子节点样本权重加和最小值sum(H)
            gamma=0,             # 节点分裂所需的最小损失函数降低值
            subsample=0.8,       # 样本随机采样做为训练集的比例
            colsample_bytree=0.8, # 使用特征比例
            objective= 'multi:softmax', # 损失函数(这里为多分类）
            num_class=4,         # 多分类问题类别数
            scale_pos_weight=1,  # 类别样本不平衡
            seed=1)

# xgboost scikit-optimize
def xgb_auto_para_tuning_bayesian(model_xgb,X,Y):
    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.80, random_state=0)
    opt = FixedBayesSearchCV(model_xgb,cv=3,n_points=2,n_jobs=4,verbose=1,
        search_spaces={
            'learning_rate': Real(0.008, 0.01),
            'max_depth': Integer(3, 10),
            'gamma':Real(0,0.5),
            'min_child_weight':Integer(1,8),
            'subsample':Real(0.6,1.0),
            'colsample_bytree':Real(0.6,1.0),
            'reg_alpha':Real(0,0.5),
            'reg_lambda':Real(0,0.5)
        },
         fit_params={
                 'eval_set': [(test_x, test_y)],
                 'eval_metric': 'mlogloss',
                 'early_stopping_rounds': 50
                 })
    opt.fit(train_x,y=train_y)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(test_x, test_y))
    print("Best parameters: ", opt.best_params_)
    print("Best estimator:", opt.best_estimator_)