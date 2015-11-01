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

class Concatenation(object):
  def __init__(self, out_lang_indexes, embeddings_layer):
    self.lang_vectors = embeddings_layer.M[out_lang_indexes]
    self.params = []

  def fprop(self, tanh_result, bias=None):
    # tanh_result.shape = (batch_size, vector_size)
    # self.lang_vectors.shape = (#languages, vector_size)
    # result.shape = (batch_size, 2 x vector_size)
    return T.concatenate([tanh_result, self.lang_vectors], axis=1)


class Embeddings(object):
  def __init__(self, vocab_size, vector_size, scale=0.08):
    self.M = sharedX(rng.randn(vocab_size, vector_size) * scale)
    self.params = [self.M]

  def fprop(self, ngram_indexes, bias=None):
    emb = self.M[ngram_indexes]
    # concatenate ngrams into one vector. the result is a matrix because it is a batch.
    return emb.reshape((ngram_indexes.shape[0], ngram_indexes.shape[1] * self.M.shape[1]))



class Attention(object):
  def __init__(self, all_lang_symbol_indexes, embeddings_layer):
    self.lang_vectors = embeddings_layer.M[all_lang_symbol_indexes]
    self.params = []

  def fprop(self, tanh_result, bias=None):
    # tanh_result.shape = (batch_size, vector_size)
    # self.lang_vectors.shape = (#languages, vector_size)
    # tanh_result_dot_langs.shape = (batch_size, #languages)
    tanh_result_dot_langs = T.dot(tanh_result, self.lang_vectors.transpose())
    # softmax_tanh_result_dot_langs.shape = (batch_size, #languages)
    self.softmax_tanh_result_dot_langs = T.nnet.softmax(tanh_result_dot_langs)
    # weighted_sum.shape = (batch_size, vector_size)
    weighted_sum = T.dot(self.softmax_tanh_result_dot_langs, self.lang_vectors)
    # result.shape = (batch_size, 2 x vector_size)
    return T.concatenate([tanh_result, weighted_sum], axis=1)
  
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
