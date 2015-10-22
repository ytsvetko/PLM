import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.utils import shuffle

rng = numpy.random.RandomState(2016)

def sharedX(X, dtype="float32"):
    return theano.shared(numpy.asarray(X, dtype=dtype))


class Activation:
    def __init__(self, func):
        self.func = func
        self.params = []

    def fprop(self, x, bias=None):
        return self.func(x)


class Linear:
    def __init__(self, vis_dim, hid_dim, scale=0.08):
        self.W = sharedX(rng.randn(vis_dim, hid_dim) * scale)
        self.b = sharedX(rng.randn(hid_dim,) * scale)
        self.params = [ self.W, self.b ]

    def fprop(self, x, bias=None):
        h = T.dot(x, self.W)+self.b
        return h

class LBL_Biased:
    def __init__(self, vis_dim, hid_dim, lang_dim, lang_activation=None, scale=0.08):
        self.W = sharedX(rng.randn(vis_dim, hid_dim) * scale)
        self.V = sharedX(rng.randn(lang_dim, hid_dim) * scale)
        self.b = sharedX(rng.randn(hid_dim,) * scale)
        self.params = [ self.W, self.V, self.b ]
        self.lang_activation = lang_activation

    def fprop(self, x, lang_dim):
        if self.lang_activation:
          h = T.dot(x, self.W)+self.lang_activation(T.dot(lang_dim, self.V)+self.b)
        else:
          h = T.dot(x, self.W)+T.dot(lang_dim, self.V)+self.b
        return h

class RNN:
    def __init__(self, vis_dim, hid_dim, scale=0.08):
        self.scale = scale
        self.hid_dim = hid_dim

        self.Wx = sharedX(rng.randn(vis_dim, hid_dim) * scale)
        self.Wh = sharedX(rng.randn(hid_dim, hid_dim) * scale)
        self.bh = sharedX(rng.randn(hid_dim,) * scale)
        self.h0 = sharedX(numpy.zeros(hid_dim,))

        self.output_info = [ self.h0 ]
        self.params = [ self.Wx, self.Wh, self.bh ]

    def fprop(self, x, bias=None):
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


if __name__ == "__main__":
  print "This is a library!"
