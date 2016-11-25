#coding: utf-8

import chainer
from chainer import computational_graph as c, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np
import argparse
from IPython import embed


from util import util
from model import auto_encoder, deep_auto_encoder, cnn_auto_encoder
from model.losser import Losser


archs = {
    'normal': auto_encoder.AutoEncoder,
    'deep': deep_auto_encoder.DeepAutoEncoder,
    'cnn': cnn_auto_encoder.CnnAutoEncoder
}


n_epoch = 10
batchsize = 32

def train(model, optimizer, x_train, y_train):
    n_train = len(x_train)
    losser = Losser(model)
    epoch_loss = []
    for epoch in range(n_epoch):
        sum_loss = np.float32(0)
        for i in range(0, n_train, batchsize):
            x_batch = Variable(x_train[i:i+batchsize])
            y_batch = Variable(y_train[i:i+batchsize])


            loss = losser(x_batch, y_batch)
            # print(loss.data)
            optimizer.zero_grads()  #backwardの直前におく！！！！！！！！！！！！！！
            loss.backward()
            optimizer.update()

            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

        sum_loss /= (i+batchsize)
        print('epoch %d done, epoch loss is %f'%(epoch, sum_loss))
        epoch_loss.append(sum_loss)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='normal',
                        help='Auto-encoder architecture')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--beta1', '-be', type=float, default=0.5,
                        help='Beta1 in Adam parameter')
    parser.add_argument('--out', '-o', default='resource',
                        help='Output directory')
    parser.add_argument('--no_dropout', action='store_true')
    parser.set_defaults(no_dropout=False)
    args = parser.parse_args()


    print('learning %s auto-encoder'%args.arch)

    save_model = '%s/%s.model'%(args.out, args.arch)
    n_epoch = args.epoch
    batchsize = args.batchsize

    #difine model and optimizer
    model = archs[args.arch]()
    if args.gpu >= 0:
        xp = cuda.cupy
        cuda.get_device(args.gpu)
        model.to_gpu()
    else:
        xp = np
        model.to_cpu()

    x_train, x_test, y_train, y_test = list(map(xp.array, util.load_mnist(noised=False))) #load mnist data

    if args.no_dropout:
        model.train = False  #without dropout
    optimizer = optimizers.Adam(alpha=0.01, beta1=args.beta1)
    optimizer.setup(model)

    model = train(model, optimizer, x_train, y_train)

    serializers.save_npz(save_model, model)


