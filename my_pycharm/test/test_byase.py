#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : test_byase.py
#Author  : ganguohua
#Time    : 2021/10/20 12:22 下午
"""
import numpy as np
from sklearn import linear_model
clf = linear_model.BayesianRidge()
ans1=[[0,0], [1, 1], [2, 2]]
y=[0, 1, 2]
clf.fit(ans1,y)
print(np.array(ans1).shape)

ans2=clf.predict([[1, 1]])
print(ans2)