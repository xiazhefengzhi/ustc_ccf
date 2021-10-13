#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : test1.py
#Author  : ganguohua
#Time    : 2021/10/8 9:15 下午
"""
import re

import pandas as pd

train_data = pd.read_csv("./base_xgb12.csv")
my_train=pd.read_csv('./xgb1.csv')
train_data['id']=my_train['id']
train_data[['id', 'isDefault']].to_csv('./base_xgb2.csv',index=False)
# print(train_data.loc[[1,2],:])

#
# def findDig(val):
#     fd = re.search('(\d+-)', val)
#
#     if fd is None:
#         return '1-' + val
#     return val + '-01'
# timeMax='1-Dec-21'
# print(pd.to_datetime(timeMax))
# train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDig))
# train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = \
#     train_data.loc[train_data[ 'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
#     years=-100)
# print(train_data['earlies_credit_mon'].value_counts())