#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : thridOliver.py
#Author  : ganguohua
#Time    : 2021/10/8 8:49 下午
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
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
train_data = pd.read_csv("../train_dataset/train_public.csv")
submit_example = pd.read_csv("./submission.csv")
test_public = pd.read_csv('../train_dataset/test_public.csv')
train_inte = pd.read_csv('../train_dataset/train_internet1.csv')
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)
null_list = []


for col in train_data.columns:
    for x in train_data[col].unique():
        if str(x) == 'nan':
            null_list.append(col)
print(null_list)


def train_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    print(oof_preds.shape)
    print(data_.shape)
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.08,
            num_leaves=2 ** 5,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,
            #             max_bin=250,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )

        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=60  # 30
                )
        temp = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        oof_preds[val_idx] = temp[:, 1]
        # 添加预测值为1的概率到sub_preds
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        # 对应不同特征的重要性
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['isDefault'] = sub_preds

    return oof_preds, test_[['loan_id', 'isDefault']], feature_importance_df


def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


#
# print(train_data['work_year'].value_counts())
# print(train_data['work_year'].describe())
# print(train_data['work_year'].unique())
def workYearDIc(x):
    if str(x) == 'nan':
        return -1
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}
timeMax = pd.to_datetime('1-Dec-21')
train_data['work_year'] = train_data['work_year'].map(workYearDIc)
test_public['work_year'] = test_public['work_year'].map(workYearDIc)
train_data['class'] = train_data['class'].map(class_dict)
test_public['class'] = test_public['class'].map(class_dict)

train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDig))
test_public['earlies_credit_mon'] = pd.to_datetime(test_public['earlies_credit_mon'].map(findDig))
train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_data.loc[train_data[
                                                                                                      'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = test_public.loc[test_public[
                                                                                                         'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])

# Internet数据处理
train_inte['work_year'] = train_inte['work_year'].map(workYearDIc)
train_inte['class'] = train_inte['class'].map(class_dict)
train_inte['earlies_credit_mon'] = pd.to_datetime(train_inte['earlies_credit_mon'])
train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])

train_data['issue_date_month'] = train_data['issue_date'].dt.month
test_public['issue_date_month'] = test_public['issue_date'].dt.month
train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
test_public['issue_date_dayofweek'] = test_public['issue_date'].dt.dayofweek

train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
test_public['earliesCreditMon'] = test_public['earlies_credit_mon'].dt.month
train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
test_public['earliesCreditYear'] = test_public['earlies_credit_mon'].dt.year

###internet数据

train_inte['issue_date_month'] = train_inte['issue_date'].dt.month
train_inte['issue_date_dayofweek'] = train_inte['issue_date'].dt.dayofweek
train_inte['earliesCreditMon'] = train_inte['earlies_credit_mon'].dt.month
train_inte['earliesCreditYear'] = train_inte['earlies_credit_mon'].dt.year

cat_cols = ['employer_type', 'industry']

from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])

    # Internet处理
    train_inte[col] = lbl.transform(train_inte[col])

# 'f1','policy_code','app_type' 这三个去掉是881
# ,'f1','policy_code','app_type'
col_to_drop = ['issue_date', 'earlies_credit_mon']
train_data = train_data.drop(col_to_drop, axis=1)
test_public = test_public.drop(col_to_drop, axis=1)

##internet处理
train_inte = train_inte.drop(col_to_drop, axis=1)
# 暂时不变
# train_inte = train_inte.rename(columns={'is_default':'isDefault'})
# data = pd.concat( [train_data,test_public] )
tr_cols = set(train_data.columns)
same_col = list(tr_cols.intersection(set(train_inte.columns)))
train_inteSame = train_inte[same_col].copy()

Inte_add_cos = list(tr_cols.difference(set(same_col)))
for col in Inte_add_cos:
    train_inteSame[col] = np.nan
y = train_data['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds, IntePre, importances = train_model(train_data, train_inteSame, y, folds)
IntePre['isDef'] = train_inte['isDefault']
from sklearn.metrics import roc_auc_score

roc_auc_score(IntePre['isDef'], IntePre.isDefault)
## 选择阈值0.05，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
InteId = IntePre.loc[IntePre.isDefault < 0.05, 'loan_id'].tolist()

train_data['dataSourse'] = 1
test_public['dataSourse'] = 1
train_inteSame['dataSourse'] = 0
train_inteSame['isDefault'] = train_inte['isDefault']
use_te = train_inteSame[train_inteSame.loan_id.isin(InteId)].copy()
data = pd.concat([train_data, test_public, use_te]).reset_index(drop=True)
# IntePre.isDefault
plt.figure(figsize=(16, 6))
plt.title("Distribution of Default values IntePre")
sns.distplot(IntePre['isDefault'], color="black", kde=True, bins=120, label='train_data')
# sns.distplot(train_inte[col],color="red", kde=True,bins=120, label='train_inte')
plt.legend();
plt.show()
train = data[data['isDefault'].notna()]

test = data[data['isDefault'].isna()]
# for col in ['sub_class', 'work_type']:
#     del train[col]
#     del test[col]
y = train['isDefault']


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
    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2021, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 30
    NBR = 10000
    VBE = 50
    test_data = xgb.DMatrix(test)
    for train_part_index, eval_index in kf.split(train_data, train_y):
        train_part = xgb.DMatrix(train_data.iloc[train_part_index], train_y.iloc[train_part_index], silent=True)
        eval = xgb.DMatrix(train_data.iloc[eval_index], train_y.iloc[eval_index])
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
        score = np.sqrt(mean_squared_error(train_y.iloc[eval_index].values, eval_pre))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    test['isDefault'] = prediction_test / 5
    test.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('./xgb1.csv', index=False)

    return


# print("-"*100)
# print(data['isDefault'].describe())
# print(pd.DataFrame(train_inte['isDefault']).value_counts())
# print(pd.DataFrame(IntePre['isDefault']).value_counts())
# print(pd.DataFrame(IntePre['isDefault']).describe())
# sum=sum([i for i in y  if i >0])
# print(len(y))
# print(sum)
# print(y.describe)
my_params = param_beyesian(train_data['isDefault'], train_data)
# # my_params={'colsample_bytree': 0.9888909225906943, 'eta': 0.054697263623566514, 'max_depth': 7.156222474025597, 'min_child_weight': 20.177979114069128, 'reg_alpha': 3.4050464815076733, 'reg_lambda': 2.476242513852993, 'subsample': 0.632056236289614}
print(my_params)
#  train_model(train, test, y, folds)
#  train_data, train_y, test_data, params
# print(train_data.describe())
print(train_data['isDefault'])
train_predict(train_data, train_data['isDefault'], test, my_params)
