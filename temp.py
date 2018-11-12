#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/11.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

df = pd.read_csv("E:\PyCharm code\Data\housing.data.txt", header=None, sep="\s+")
# print(df.head(50))

x = df.iloc[:, :-1].values
y = df.iloc[:, 13].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

"""
    la = Lasso()
    la.fit(x_train, y_train)
    y_test_pre = la.predict(x_test)
    y_train_pre = la.predict(x_train)
    
    r2_test = r2_score(y_test_pre, y_test)
    r2_train = r2_score(y_train_pre, y_train)
    
    print(la.coef_)
    print("r2_score of train : {r2: .2f}".format(r2 = r2_train))
    print("r2_score of test : {r2: .2f}".format(r2 = r2_test))
"""
temps = np.linspace(0, 100, 100)
R2_test = []
R2_train = []
for temp in temps:
    ri = Ridge(alpha=temp)
    ri.fit(x_train, y_train)
    y_test_pre = ri.predict(x_test)
    y_train_pre = ri.predict(x_train)

    r2_test = r2_score(y_test_pre, y_test)
    r2_train = r2_score(y_train_pre, y_train)
    R2_train.append(r2_train)
    R2_test.append(r2_test)
plt.plot(temps, R2_test)
plt.plot(temps, R2_train)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()




