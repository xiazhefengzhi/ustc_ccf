#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : test_STACK.py
#Author  : ganguohua
#Time    : 2021/10/14 9:57 上午
"""
import numpy as np
a=np.arange(12)
b=np.arange(12)
print(a.shape)
ans1=np.hstack((a,b))
print(ans1.shape)
