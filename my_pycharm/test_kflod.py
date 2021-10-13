#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : test_kflod.py
#Author  : ganguohua
#Time    : 2021/10/9 8:40 上午
"""
from sklearn.model_selection import  KFold
import  numpy as np
X=np.arange(20).reshape(10,2)
y=np.random.choice([1,0],10,p=[0.4,.6])
folds=KFold(n_splits=5,shuffle=True,random_state=9527)
for my_x, my_y in folds.split(X, y):
    print(my_x)
    print(my_y)
# for index,(x,y) in enumerate(folds.split(X)):
#     print(index)
#     print(x,y)