#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/28.


from IPython.display import Image
import pandas as pd


df = pd.read_csv("E:\PyCharm code\Data\housing.data.txt", header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


# ## Visualizing the important characteristics of a dataset

import matplotlib.pyplot as plt
import seaborn as sns

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('img.png', dpi=300)
plt.show()


import numpy as np

cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.show()


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

from sklearn.preprocessing import StandardScaler

# 取X值使用两个[]为了将一维数据转为二维数据，转为n行1列
X = df[['RM']].values
y = df['MEDV'].values

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
# 转为二维数据进行标准化，然后调用flatten再转为一维
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()


lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# 显示每次训练和对应的cost值
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# 散点图是原始数据，直线式拟合后的结果
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

# 由于只有一个变量，下面参数分别对应直线的斜率slope和截距Intercept
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


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




