#coding: utf-8

import chainer
from chainer import computational_graph as c, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np
from IPython import embed

from util import util



n_input = 28

class CnnAutoEncoder(chainer.Chain):
    train = True
    def __init__(self):
        super(CnnAutoEncoder, self).__init__(
            cnv1 = L.Convolution2D(None, 16, 3, pad=1),
            #cnv2 = L.Convolution2D(None, 8, 3, pad=1),
            #cnv3 = L.Convolution2D(None, 8, 3, pad=1),
            #decnv3 = L.Convolution2D(None, 8, 3, pad=1),
            #decnv2 = L.Convolution2D(None, 8, 3, pad=1),
            decnv1 = L.Convolution2D(None, 16, 3, pad=1),
            decoder = L.Convolution2D(None, 1, 3, pad=1),
        )

    def __call__(self, x, hidden=False):
        batchsize = x.data.shape[0]
        x = F.reshape(x, (batchsize, 1, n_input, n_input))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.cnv1(x))), 2)
        #h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.cnv2(h))), 2, stride=2)
        #y = F.max_pooling_2d(F.local_response_normalization(F.relu(self.cnv3(h))), 2, stride=2)

        if hidden:
            return y

        #h = F.unpooling_2d(F.local_response_normalization(F.relu(self.decnv3(y))), 2, stride=2, cover_all=False)
        #h = F.unpooling_2d(F.local_response_normalization(F.relu(self.decnv2(h))), 2, stride=2, cover_all=False)
        #h = F.unpooling_2d(F.local_response_normalization(F.relu(self.decnv1(h))), 2, cover_all=False)
        h = F.local_response_normalization(F.relu(self.decnv1(F.unpooling_2d(h, 2, cover_all=False)))) #上のやつコピーして追加した
        x_hat = F.sigmoid(self.decoder(h))
        x_hat = F.reshape(x_hat, (batchsize, n_input*n_input))
        util.draw_graph(x_hat)
        return x_hat

# class CnnBNAutoEncoder(chainer.Chain):
#     train = True
#     def __init__(self):
#         super(CnnBNAutoEncoder, self).__init__(
#             cnv1 = L.Convolution2D(None, 16, 3, pad=1),
#             cnv2 = L.Convolution2D(None, 8, 3, pad=1),
#             cnv3 = L.Convolution2D(None, 8, 3, pad=1),
#             bn1 = L.BatchNormalization(None),
#             bn2 = L.BatchNormalization(None),
#             bn3 = L.BatchNormalization(None),
#             decnv3 = L.Convolution2D(None, 8, 3, pad=1),
#             decnv2 = L.Convolution2D(None, 8, 3, pad=1),
#             decnv1 = L.Convolution2D(None, 16, 3),
#             decoder = L.Convolution2D(None, 1, 3, pad=1),
#         )
#
#     def __call__(self, x, hidden=False):
#         x = F.reshape(x, (x.data.shape[0], 1, n_input, n_input))
#         h = F.relu(self.bn1(self.cnv1(x), test=not self.train))
#         h = F.relu(self.bn2(self.cnv2(h), test=not self.train))
#         y = F.relu(self.bn3(self.cnv3(h), test=not self.train))
#
#         if hidden:
#             return y
#
#         h = F.unpooling_2d(F.relu(self.decnv3(y)), 2, stride=2, cover_all=False)
#         h = F.unpooling_2d(F.relu(self.decnv2(h)), 2, stride=2, cover_all=False)
#         h = F.unpooling_2d(F.relu(self.decnv1(h)), 2, stride=2, cover_all=False)
#         x_hat = F.sigmoid(self.decoder(h))
#         x_hat = F.reshape(x_hat, (32, n_input*n_input))
#         return x_hat
#
