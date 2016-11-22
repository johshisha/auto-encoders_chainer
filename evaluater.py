#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np
import random

from util import util
from model.auto_encoder import AutoEncoder

saved_model = 'resource/auto_encoder.model'

x_train, x_test, y_train, y_test = util.load_mnist(N=10000) #load mnist data

model = AutoEncoder()
serializers.load_npz(saved_model, model)
model.train = False


#show test sample estimation
n_sample = 32
indexs = random.sample(range(len(y_test)), n_sample)#抽出する添字を取得
x, t = Variable(x_test[indexs]), Variable(y_test[indexs])
y = model(x if n_sample != 1 else F.reshape(x, (n_sample, x.data.shape[0])))
util.draw_digits(y, t)

