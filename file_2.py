#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/21.
#   模型评估方法---流出法  交叉验证法  自助法

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("E:\PyCharm code\Data\iris.data.txt", header=None)

y = df.iloc[:, 4].values
x = df.iloc[:, [0, 1, 2, 3]].values

le = LabelEncoder()
le.fit(y)
y_i = le.transform(y)
"""
#   流出法   
#
from sklearn.model_selection import train_test_split
n = 10
score = 0.0
for i in range(n):
    x_train, x_test, y_train, y_test = train_test_split(x, y_i, test_size=0.3, stratify=y)

    ss = StandardScaler()
    ss.fit(x_train)
    x_train_std = ss.transform(x_train)
    x_test_std = ss.transform(x_test)

    lr = LogisticRegression(C=100, random_state=1)
    lr.fit(x_train_std, y_train)
    s = lr.score(x_test_std, y_test)
    score = score+s
    print('%d -- Accuracy is %.2f' % (i+1, s))
print('Average accuracy is %.2f' % (score/n))
"""

#   交叉验证法
from sklearn.model_selection import StratifiedKFold
kfolk = StratifiedKFold(n_splits=10, random_state=1).split(x, y_i)

scores = []
lr = LogisticRegression(C=100, random_state=1)

for k, (train, test) in enumerate(kfolk):
    lr.fit(x[train], y_i[train])
    score = lr.score(x[test], y_i[test])
    scores.append(score)
    print('Ford : %d , Acc : %.3f' % (k+1, score))
print('\nAverage acc is : %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))












