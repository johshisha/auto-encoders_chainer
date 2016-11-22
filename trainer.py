#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np


from util import util
from model.auto_encoder import AutoEncoder
from model.losser import Losser



n_epoch = 10
batchsize = 32

save_model = 'resource/auto_encoder.model'

x_train, x_test, y_train, y_test = util.load_mnist(noised=False) #load mnist data


#difine model and optimizer
model = AutoEncoder()
optimizer = optimizers.Adam()
optimizer.setup(model)
losser = Losser(model)


n_train = len(x_train)

epoch_loss = []
for epoch in range(n_epoch):
    sum_loss = np.float32(0)
    for i in range(0, n_train, batchsize):
        x_batch = Variable(x_train[i:i+batchsize])
        y_batch = Variable(y_train[i:i+batchsize])

        optimizer.zero_grads()
        loss = losser(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

    sum_loss /= (i+batchsize)
    print('epoch %d done, epoch loss is %f'%(epoch, sum_loss))
    epoch_loss.append(sum_loss)

serializers.save_npz(save_model, model)


