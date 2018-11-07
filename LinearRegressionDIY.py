#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/11.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class LinearRegressionDIY(object):

    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter

    def calculate(self, x):
        return np.dot(x, self.w[1:])+self.w[0]

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1]+1)
        self.costs = []
        for i in range(self.n_iter):
            temp = self.calculate(x)
            error = y-temp
            self.w[1:] += self.eta*np.dot(x.T, error)
            self.w[0] += self.eta*np.sum(error)
            cost = 0.5*(error**2).sum()
            self.costs.append(cost)
        return self

    def predict(self, x):
        return self.calculate(x)

df = pd.read_csv("E:\PyCharm code\Data\housing.data.txt", header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
X = df[['RM']].values
y = df['MEDV'].values

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionDIY(eta=0.001, n_iter=20)
lr.fit(X_std, y_std)


# 散点图是原始数据，直线式拟合后的结果
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()




# 输入一个特征值5，并进行标准化
num_rooms_std = sc_x.transform(np.array([[5.0]]))
# 使用训练好的模型进行预测,预测结果也是标准化的数据
price_std = lr.predict(num_rooms_std)
print(price_std)
# 通过inverse_transform反向操作，将标准化值转为真实的价格数据
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))


# 使用python模型库的线性回归模型做一样的事情
# ## Estimating the coefficient of a regression model via scikit-learn

from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()




















