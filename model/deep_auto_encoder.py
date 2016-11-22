#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np


n_input = 784

class DeepAutoEncoder(chainer.Chain):
    train = True
    def __init__(self):
        super(DeepAutoEncoder, self).__init__(
            e1 = L.Linear(n_input, 128),
            e2 = L.Linear(128, 64),
            e3 = L.Linear(64, 32),
            d3 = L.Linear(32, 64),
            d2 = L.Linear(64, 128),
            d1 = L.Linear(128, n_input),
        )

    def __call__(self, x, hidden=False):
        h = F.dropout(F.relu(self.e1(x)), train=self.train)
        h = F.dropout(F.relu(self.e2(h)), train=self.train)
        y = F.dropout(F.relu(self.e3(h)), train=self.train)
        if hidden:
            return y
        h = F.dropout(F.relu(self.d3(y)), train=self.train)
        h = F.dropout(F.relu(self.d2(h)), train=self.train)
        x_hat  = F.dropout(F.sigmoid(self.d1(h)),  train=self.train)
        return x_hat

