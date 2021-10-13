#!/usr/bin/env python
# coding: utf-8


import warnings

import numpy as np
import pandas as pd
import xgboost as  xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

train_bank = pd.read_csv('../train_dataset/train_public.csv')
train_internet = pd.read_csv('../train_dataset/train_internet1.csv')
test = pd.read_csv('../train_dataset/test_public.csv')

# ### 数据预处理

# In[3]:


common_cols = []
for col in train_bank.columns:
    if col in train_internet.columns:
        common_cols.append(col)
    else:
        continue
len(common_cols)

# In[4]:


print(len(train_bank.columns))
print(len(train_internet.columns))

# In[5]:


train_bank_left = list(set(list(train_bank.columns)) - set(common_cols))
train_internet_left = list(set(list(train_internet.columns)) - set(common_cols))

train_bank_left

# In[6]:


train_internet_left

# In[7]:


train1_data = train_internet[common_cols]
train2_data = train_bank[common_cols]
test_data = test[common_cols[:-1]]


# In[8]:


import datetime

# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train1_data['issue_date'] = pd.to_datetime(train1_data['issue_date'])
# 提取多尺度特征
train1_data['issue_date_y'] = train1_data['issue_date'].dt.year
train1_data['issue_date_m'] = train1_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train1_data['issue_date_diff'] = train1_data['issue_date'].apply(lambda x: x - base_time).dt.days
train1_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train1_data.drop('issue_date', axis=1, inplace=True)

# In[9]:


# 日期类型：issueDate，earliesCreditLine
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
train2_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train2_data.drop('issue_date', axis=1, inplace=True)
train2_data

# In[10]:


employer_type = train1_data['employer_type'].value_counts().index
industry = train1_data['industry'].value_counts().index

# In[11]:


emp_type_dict = dict(zip(employer_type, [0, 1, 2, 3, 4, 5]))
industry_dict = dict(zip(industry, [i for i in range(15)]))

# In[12]:


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

# In[13]:


# 日期类型：issueDate，earliesCreditLine
# train[cat_features]
# 转换为pandas中的日期类型
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

# ## 模型使用
# 1) LigthGBM
# 2) NN

# ##### 使用internet和bank数据共同特征总量训练

# In[16]:


X_train1 = train1_data.drop(['isDefault', 'earlies_credit_mon', 'loan_id', 'user_id'], axis=1, inplace=False)
y_train1 = train1_data['isDefault']

X_train2 = train2_data.drop(['isDefault', 'earlies_credit_mon', 'loan_id', 'user_id'], axis=1, inplace=False)
y_train2 = train2_data['isDefault']

X_train = pd.concat([X_train1, X_train2])
y_train = pd.concat([y_train1, y_train2])
y_train=pd.DataFrame(y_train)
y_train.set_index('isDefault')
print(y_train.columns)
X_test = test_data.drop(['earlies_credit_mon', 'loan_id', 'user_id'], axis=1, inplace=False)


# # 利用Internet数据预训练模型1
# clf_ex = lightgbm.LGBMRegressor(n_estimators=200)
# clf_ex.fit(X=X_train, y=y_train)
# clf_ex.booster_.save_model('LGBMmode.txt')
# pred = clf_ex.predict(X_test)

def params_append(params):
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'
    params['min_child_weight'] = int(params['min_child_weight'])
    params['max_depth'] = int(params['max_depth'])
    return params


def param_beyesian(train_y, train_data):
    # feats = [f for f in train_data.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    train_data = xgb.DMatrix(train_data, train_y,silent=True)

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
    print(test.to_csv('./base_xgb12.csv'))
    # test.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('./base_xgb2.csv', index=False)

    return
# {'colsample_bytree': 0.5292327549468065, 'eta': 0.025042320369886798, 'max_depth': 5.183528248675014, 'min_child_weight': 1.4563747204930961, 'reg_alpha': 0.09410187284743576, 'reg_lambda': 4.100561277776243, 'subsample': 0.8922799985675434}
#
my_params = {'colsample_bytree': 0.5292327549468065, 'eta': 0.025042320369886798, 'max_depth': 5.183528248675014, 'min_child_weight': 1.4563747204930961, 'reg_alpha': 0.09410187284743576, 'reg_lambda': 4.100561277776243, 'subsample': 0.8922799985675434}
# print((X_test).columns)
train_predict(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), my_params)
