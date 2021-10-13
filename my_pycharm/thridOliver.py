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
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
        temp=clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
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
# print(train_data['work_year'].value_counts())
test_public['work_year'] = test_public['work_year'].map(workYearDIc)
# print(train_data['class'].unique())
train_data['class'] = train_data['class'].map(class_dict)
test_public['class'] = test_public['class'].map(class_dict)


# print(train_data.head(10))
train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDig))
test_public['earlies_credit_mon'] = pd.to_datetime(test_public['earlies_credit_mon'].map(findDig))
train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_data.loc[train_data[                                                                                                      'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = test_public.loc[test_public[
                                                                                                         'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])

# Internet数据处理
train_inte['work_year'] = train_inte['work_year'].map(workYearDIc)
train_inte['class'] = train_inte['class'].map(class_dict)
### 不需要进行数据处理操作earlies_credit_mon ，internet
train_inte['earlies_credit_mon'] = pd.to_datetime(train_inte['earlies_credit_mon'])
train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])

train_data['issue_date_month'] = train_data['issue_date'].dt.month
test_public['issue_date_month'] = test_public['issue_date'].dt.month
train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
cat_cols = ['employer_type', 'industry']

from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])

    # Internet处理
    train_inte[col] = lbl.transform(train_inte[col])
    # 进行打印之前的映射关系
    # print(lbl.classes_)

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
feats = [f for f in same_col if f not in ['isDefault']]
test_inteSame = test_public[feats].copy()
# for col in Inte_add_cos:
#     train_inteSame.drop([col],inplace=True)

# 81后加
# for col in cat_cols:
#     dum = pd.get_dummies(data[col], prefix='OneHot_'+col +'_')
#     data = pd.concat([data, dum], axis=1)
# #     del data[col]
#     del dum
y = train_data['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=9527)
print(len(train_inteSame.columns))
print(len(test_inteSame.columns))
oof_preds, IntePre, importances = train_model(train_inteSame, test_inteSame, train_inteSame['isDefault'], folds)
IntePre.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('nn3.csv', index=False)

IntePre['isDef'] = train_inte['isDefault']




# # 选择阈值0.05，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
# InteId = IntePre.loc[IntePre['isDefault']<0.05, 'loan_id'].tolist()
# print(len(InteId))
# train_data['dataSourse'] = 1
# test_public['dataSourse'] = 1
# train_inteSame['dataSourse'] = 0
# train_inteSame['isDefault'] = train_inte['isDefault']
# use_te = train_inteSame[train_inteSame.loan_id.isin( InteId )].copy()
# data = pd.concat([ train_data,test_public,use_te]).reset_index(drop=True)
# train = data[data['isDefault'].notna()]
# test  = data[data['isDefault'].isna()]
#
# y = train['isDefault']
# folds = KFold(n_splits=5, shuffle=True, random_state=546789)
# oof_preds, test_preds, importances = train_model(train, test, y, folds)
# test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('nn2.csv', index=False)
# display_importances(importances)