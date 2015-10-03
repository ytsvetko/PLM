from collections import OrderedDict

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.utils import shuffle

rng = numpy.random.RandomState(42)
trng = RandomStreams(42)

def sharedX(X, dtype="float32"):
    return theano.shared(numpy.asarray(X, dtype=dtype))


class Activation:
    def __init__(self, func):
        self.func = func
        self.params = []

    def fprop(self, x):
        return self.func(x)


class Linear:
    def __init__(self, vis_dim, hid_dim, scale):
        self.W = sharedX(rng.randn(vis_dim, hid_dim) * scale)
        self.b = sharedX(rng.randn(hid_dim,) * scale)
        self.params = [ self.W, self.b ]

    def fprop(self, x):
        h = T.dot(x, self.W)+self.b
        return h


class RNN:
    def __init__(self, vis_dim, hid_dim, scale):
        self.scale = scale
        self.hid_dim = hid_dim

        self.Wx = sharedX(rng.randn(vis_dim, hid_dim) * scale)
        self.Wh = sharedX(rng.randn(hid_dim, hid_dim) * scale)
        self.bh = sharedX(rng.randn(hid_dim,) * scale)
        self.h0 = sharedX(numpy.zeros(hid_dim,))

        self.output_info = [ self.h0 ]
        self.params = [ self.Wx, self.Wh, self.bh ]

    def fprop(self, x):
        def step(u_t, h_tm1):
            h = T.nnet.sigmoid(T.dot(u_t,self.Wx) + T.dot(h_tm1,self.Wh) + self.bh)
            return h

        h, _ = theano.scan(
            fn = step,
            sequences = x,
            outputs_info = self.output_info,
            n_steps=x.shape[0]
            )
        return h


def sgd(cost, params, lr):
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        updates[param] = param - lr * gparam
    return updates


def prop(layers, x):
    for i, layer in enumerate(layers):
        if i == 0:
            layer_out = layer.fprop(x)
        else:
            layer_out = layer.fprop(layer_out)
    return layer_out


def get_params(layers):
    params = []
    for layer in layers:
        params += layer.params
    return params


def TrainRNNLM(train_x, train_y, out_dim = 69, hidden_dim = 20, batch_size=100, epochs=50):
    x, t = T.matrix("x"), T.lvector("t")

    print("train_x.shape", train_x.shape)
    print("train_y.shape", train_y.shape)
    print("vocab size", out_dim)
    layers = [
                RNN(train_x.shape[1], hidden_dim, scale=0.08),
                Linear(hidden_dim, out_dim, scale=0.08),
                Activation(T.nnet.softmax),
            ]

    prob = prop(layers, x) 
    cost = - T.mean((T.log(prob))[T.arange(x.shape[0]), t])

    ## Collect Parameters
    params = get_params(layers) 

    ## Define update graph
    updates = sgd(cost, params, lr=0.01) 

    ## Compile Function
    train = theano.function([x, t], cost, updates=updates, 
                        allow_input_downcast=True)
        
    ## Train
    nbatches = train_x.shape[0]//batch_size
    for epoch in range(epochs):
      train_x, train_y = shuffle(train_x, train_y)  # Shuffle Samples !!
      for i in range(nbatches):
        start = i * batch_size
        end = start + batch_size
        cost = train(train_x[start:end], train_y[start:end])
        if i % 1000 == 0:
          print "EPOCH:: %i, Iteration %i, cost: %.3f"%(epoch+1, i, cost)
        
if __name__ == "__main__":
  print "This is a library!"
