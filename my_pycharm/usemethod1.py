#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : usemethod1.py
#Author  : ganguohua
#Time    : 2021/10/7 3:51 下午
"""
import datetime
import warnings
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import xgboost as  xgb
from sklearn.metrics import mean_squared_error
import lightgbm
import numpy as np
import pandas as pd
from sklearn.decomposition import  PCA
from sklearn.preprocessing import  StandardScaler
warnings.filterwarnings('ignore')
train_bank = pd.read_csv('../train_dataset/train_public.csv')
train_internet = pd.read_csv('../train_dataset/train_internet.csv')
test = pd.read_csv('../train_dataset/test_public.csv')
common_cols = []
for col in train_bank.columns:
    if col in train_internet.columns:
        common_cols.append(col)
    else:
        continue
print(common_cols)
train_bank_left = list(set(list(train_bank.columns)) - set(common_cols))
train_internet_left = list(set(list(train_internet.columns)) - set(common_cols))
train1_data = train_internet[common_cols]
train2_data = train_bank[common_cols]
### 不包括最后一行
test_data = test[common_cols[:-1]]
# print(train1_data['issue_date'].iloc[:10])
train1_data['issue_date'] = pd.to_datetime(train1_data['issue_date'])
train1_data['issue_date_y'] = train1_data['issue_date'].dt.year
train1_data['issue_date_m'] = train1_data['issue_date'].dt.month
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
train1_data['issue_date_diff'] = train1_data['issue_date'].apply(lambda x: x - base_time).dt.days
train1_data.drop('issue_date', axis=1, inplace=True)
# 转换为pandas中的日期类型
train2_data['issue_date'] = pd.to_datetime(train2_data['issue_date'])
# 提取多尺度特征
train2_data['issue_date_y'] = train2_data['issue_date'].dt.year
train2_data['issue_date_m'] = train2_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train2_data['issue_date_diff'] = train2_data['issue_date'].apply(lambda x: x - base_time).dt.days
train2_data.drop('issue_date', axis=1, inplace=True)
### 将数值类数据转化成index
employer_type = train1_data['employer_type'].value_counts().index
industry = train1_data['industry'].value_counts().index
emp_type_dict = dict(zip(employer_type, [0, 1, 2, 3, 4, 5]))
industry_dict = dict(zip(industry, [i for i in range(15)]))
# todo  统计需要填充的数据的占比
train1_data['work_year'].fillna('10+ years', inplace=True)
train2_data['work_year'].fillna('10+ years', inplace=True)
work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
                 '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
train1_data['work_year'] = train1_data['work_year'].map(work_year_map)
train2_data['work_year'] = train2_data['work_year'].map(work_year_map)

train1_data['class'] = train1_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
train2_data['class'] = train2_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

train1_data['employer_type'] = train1_data['employer_type'].map(emp_type_dict)
train2_data['employer_type'] = train2_data['employer_type'].map(emp_type_dict)

train1_data['industry'] = train1_data['industry'].map(industry_dict)
train2_data['industry'] = train2_data['industry'].map(industry_dict)
print(train1_data.head(10))
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])
# 提取多尺度特征
test_data['issue_date_y'] = test_data['issue_date'].dt.year
test_data['issue_date_m'] = test_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
test_data['issue_date_diff'] = test_data['issue_date'].apply(lambda x: x - base_time).dt.days
test_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
test_data.drop('issue_date', axis=1, inplace=True)
test_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
                 '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
test_data['work_year'] = test_data['work_year'].map(work_year_map)
test_data['class'] = test_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
test_data['employer_type'] = test_data['employer_type'].map(emp_type_dict)
test_data['industry'] = test_data['industry'].map(industry_dict)

X_train1 = train1_data.drop(['is_default', 'earlies_credit_mon', 'loan_id', 'user_id'], axis=1, inplace=False)
y_train1 = train1_data['is_default']

X_train2 = train2_data.drop(['is_default', 'earlies_credit_mon', 'loan_id', 'user_id'], axis=1, inplace=False)
y_train2 = train2_data['is_default']

X_train = pd.concat([X_train1, X_train2])
y_train = pd.concat([y_train1, y_train2])

#  添加pca降维操作
my_std=StandardScaler()
X_std=my_std.fit_transform(X_train)
X_std[np.isnan(X_std)] = 0
print(X_std[:20])
pca = PCA(n_components = 16)
pca.fit_transform(X_std)
print(pca.singular_values_)
print(X_std.shape)

X_test = test_data.drop(['earlies_credit_mon', 'loan_id', 'user_id'], axis=1, inplace=False)

# 利用Internet数据预训练模型1
clf_ex = lightgbm.LGBMRegressor(n_estimators=200)
clf_ex.fit(X=X_std, y=y_train)
clf_ex.booster_.save_model('LGBMmode.txt')
my_std.transform(X_test)
X_test[np.isnan(X_test)] = 0
pca.transform(X_test)
pred = clf_ex.predict(X_test)
pred = [1 if i > 0 else 0 for i in pred]
submission = pd.DataFrame({'id': test['loan_id'], 'isDefault': pred})
submission.to_csv('submission.csv', index=None)

