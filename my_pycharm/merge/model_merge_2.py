#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : model_merge_2.py
#Author  : ganguohua
#Time    : 2021/10/13 8:14 下午
"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : baseline_89_2.py
#Author  : ganguohua
#Time    : 2021/10/13 4:28 下午
"""
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import StratifiedKFold
from dateutil.relativedelta import relativedelta

train_data = pd.read_csv('../nw_train_public.csv')
submit_example = pd.read_csv('../../train_dataset/submit_example.csv')
test_public = pd.read_csv('../../train_dataset/test_public.csv')
train_inte = pd.read_csv('../../train_dataset/train_internet1.csv')

pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)


def train_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
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
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 30
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

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
print(len(train_data),train_data.columns.values)
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
# folds = KFold(n_splits=5, shuffle=True, random_state=546789)
# oof_preds, IntePre, importances = train_model(train_data, train_inteSame, y, folds)
#  使用不同的模型然后进行融合


print(len(train_inteSame.columns.values),train_data.columns.values)
print(len(train_inteSame.columns.values),train_inteSame.columns.values)
##  目前 train_data中的字段和train_inteSame 中的字段是一样的
##  默认 lightgbm 会使用中值填充空值
##  添加过滤某些字段的操作
## df.isnull().any()
feats = [f for f in train_data.columns if f not in ['known_outstanding_loan', 'known_dero', 'app_type','loan_id', 'user_id', 'isDefault']]
## todo  进行优化进行中值填充的方法

for i in feats:
    train_data[i]  =  train_data[i].fillna(value=train_data[i].mean())
    train_inteSame[i]=train_inteSame[i].fillna(value=train_inteSame[i].mean())
# null_list=['known_outstanding_loan', 'known_dero', 'app_type']
# train_inteSame[null_list]=train_inteSame[null_list].fillna(value=0)
print(len(set(train_data.columns.values)&set(train_inteSame.columns.values)))
print(train_data[feats].isnull().any())
## 使用随机森林 进行数据的预测

from sklearn.ensemble import  RandomForestClassifier
rf=RandomForestClassifier(max_features='auto',oob_score=True,random_state=1,n_jobs=-1)
rf.fit(X=train_data[feats],y=y)
ans1=rf.predict(train_inteSame[feats])
print(roc_auc_score(train_inteSame['isDefault'],ans1))
## 使用xgboost 进行数据的预测
import  xgboost as xgb
params={'eta':0.01 ,'max_depth':11,'objective':'reg:liner','eval_metric':'rmse'}
dtrain=xgb.DMatrix(data=train_data[feats],label=y)
dtest=xgb.DMatrix(data=train_inteSame[feats],label=train_inteSame['isDefault'])





# InteId = IntePre.loc[IntePre.isDefault<0.05, 'loan_id'].tolist()
# train_inte = train_inte.rename(columns={'is_default':'isDefault'})

# train_data['dataSourse'] = 1
# test_public['dataSourse'] = 1
# train_inte['dataSourse'] = 0


# use_te = train_inte[train_inte.loan_id.isin( InteId )].copy()
# data = pd.concat([ train_data,test_public,use_te]).reset_index(drop=True)

# IntePre.isDefault
# plt.figure(figsize=(16, 6))
# plt.title("Distribution of Default values IntePre")
# sns.distplot(IntePre['isDefault'], color="black", kde=True, bins=120, label='train_data')
# # sns.distplot(train_inte[col],color="red", kde=True,bins=120, label='train_inte')
# plt.legend();
# plt.show()
# train = data[data['isDefault'].notna()]
# test = data[data['isDefault'].isna()]
#
# del data
# del train_data, test_public
#
# y = train['isDefault']
# folds = KFold(n_splits=5, shuffle=True, random_state=546789)
# oof_preds, test_preds, importances = train_model(train, test, y, folds)
# test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('baseline891.csv', index=False)