#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np


n_input = 784
n_units = 128

class AutoEncoder(chainer.Chain):
    train = True
    def __init__(self):
        super(AutoEncoder, self).__init__(
            encoder = L.Linear(n_input, n_units),
            decoder = L.Linear(n_units, n_input),
        )

    def __call__(self, x, hidden=False):
        h = F.dropout(F.relu(self.encoder(x)), train=self.train)
        if hidden:
            return h
        x_hat  = F.dropout(F.sigmoid(self.decoder(h)),  train=self.train)
        return x_hat

