#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/11.

import numpy as np


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
    
















