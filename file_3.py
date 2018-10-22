#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/22.
# 性能度量
# 混淆矩阵，准确度，精确度，召回率，F1分数，特异度

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("E:\PyCharm code\Data\iris.data.txt", header=None)

y = df.iloc[-100:, 4].values
x = df.iloc[-100:, [0, 1, 2, 3]].values

le = LabelEncoder()
le.fit(y)
y_i = le.transform(y)

kfold = StratifiedKFold(n_splits=10, random_state=1).split(x, y_i)

lr = LogisticRegression(C=100.0, random_state=1)

for k, (train, test) in enumerate(kfold):

    lr.fit(x[train], y_i[train])
    y_pred = lr.predict(x[test])

    conf_mat = confusion_matrix(y_true=y_i[test], y_pred=y_pred)
    # 准确度
    score = lr.score(x[test], y_i[test])
    print("Acc is : %.3f" % (score))
    # 精准度
    precision = precision_score(y_true=y_i[test], y_pred=y_pred)
    print("Precision is : %.3f" % (precision))
    # 召回率/敏感度
    recall = recall_score(y_true=y_i[test], y_pred=y_pred)
    print("Recall is : %.3f" % (recall))
    # F1分数
    f1 = f1_score(y_true=y_i[test], y_pred=y_pred)
    print("F1_score is : %.3f" % (f1))
    # 特异度/真负率/真阴性率
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    spec = float(tn) / (float(tn) + float(fp))
    print('Spec: %.3f' % spec)
    print("\n")












