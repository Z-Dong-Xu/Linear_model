#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/11.

import numpy as np


class LinearRegressionDIY(object):

    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        w = np.zeros(x.shape[1]+1)
        cost = []
        for i in range(self.n_iter):
            temp = np.dot(x, w.T)


    def calculate(self, x):
        pass
        return res












