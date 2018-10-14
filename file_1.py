#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/11.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("E:\PyCharm code\Data\iris.data.txt", header=None)
# print(df.head(10))
# print(df.tail(10))
# print(type(df))

y = df.iloc[:, 4].values
# print(len(y), np.unique(y))

le = LabelEncoder()
y_i = le.fit_transform(y)
# print(np.unique(y_i))

x = df.iloc[:, [0, 1, 2, 3]].values
# print(type(x))

x_train, x_test, y_train, y_test = train_test_split(x, y_i, test_size=0.3, random_state=1, stratify=y)
# print(len(x_train), len(x_test))

# print(type(y_train), np.shape(y_train), type(y_i), np.shape(y_i))

# for i in np.unique(y_train):
#     print(i, list(y_train).count(i))

ss = StandardScaler()
ss.fit(x_train)

x_train_std = ss.transform(x_train)
x_test_std = ss.transform(x_test)
# print(x_train_std[:10])

X = pd.DataFrame(x_test_std)
Y = pd.DataFrame(y_test)
STD = pd.concat([X, Y], axis=1)
# print(STD[:10])
STD.to_csv('E:\PyCharm code\Data\STDiris.txt', sep=',', header=None, index=None)

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(x_train_std, y_train)
print("finish")

print(lr.predict([[0.29247235179042536, -0.6273171437763873, 0.11738784104063459, 0.1358648172970528]]))













