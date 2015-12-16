import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.utils import shuffle

rng = numpy.random.RandomState(2016)
floatX = theano.config.floatX
def sharedX(X, dtype="float32", name=None):
  return theano.shared(numpy.asarray(X, dtype=dtype), name=name)


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
    self.M = sharedX(rng.randn(vocab_size, vector_size) * scale, name="Embeddings.M")
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
    self.W = sharedX(rng.randn(vis_dim, hid_dim) * scale, name="Linear.W")
    self.b = sharedX(numpy.zeros(hid_dim,), name="Linear.b")
    self.params = [ self.W, self.b ]

  def fprop(self, x, bias=None):
    h = T.dot(x, self.W)+self.b
    return h

class Linear_Biased:
  def __init__(self, vis_dim, hid_dim, lang_vec, lang_dim=30, lang_activation=T.nnet.sigmoid, scale=0.08):
    self.lang_vec = lang_vec
    self.L = sharedX(rng.randn(186, lang_dim) * scale, name="Linear_Biased.L") # dimension reduction for linguistic vector
    self.l_b = sharedX(numpy.zeros(lang_dim,), name="Linear_Biased.l_b")

    self.W = sharedX(rng.randn(vis_dim, hid_dim) * scale, name="Linear_Biased.W")
    self.V = sharedX(rng.randn(lang_dim, hid_dim) * scale, name="Linear_Biased.V")
    self.b = sharedX(numpy.zeros(hid_dim,), name="Linear_Biased.b")

    self.params = [ self.L, self.l_b, self.W, self.V, self.b ]
    self.lang_activation = lang_activation

  def fprop(self, x):
    h_lang = self.lang_activation(T.dot(self.lang_vec, self.L)+self.l_b)
    h = T.dot(x, self.W)+T.dot(h_lang, self.V)+self.b
    return h

class RNN:
  def __init__(self, vis_dim, hid_dim, with_batch=True, scale=0.08):
      self.scale = scale
      self.hid_dim = hid_dim

      self.Wx = sharedX(rng.randn(vis_dim, hid_dim) * scale, name="RNN.Wx")
      self.Wh = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="RNN.Wh")
      self.bh = sharedX(numpy.zeros(hid_dim,), name="RNN.bh")
      self.h0 = sharedX(numpy.zeros(hid_dim,), name="RNN.h0")

      self.output_info = [ self.h0 ]
      self.params = [ self.Wx, self.Wh, self.bh, self.h0 ]
      self.with_batch = with_batch

  def fprop(self, x, bias=None):
      def recurrence(u_t, h_tm1):
          h = T.tanh(T.dot(u_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
          return h

      h, _ = theano.scan(
          fn = recurrence,
          sequences = x,
          outputs_info = self.output_info,
          n_steps=x.shape[0]
          )
      return h

class LSTM(object):
  """
  Recurrent neural network. Can be used with or without batches.
  Without batches:
      Input: matrix of dimension (sequence_length, vis_dim)
      Output: vector of dimension (output_dim)
  With batches:
      Input: tensor3 of dimension (sequence_length, batch_size, vis_dim)
      Output: matrix of dimension (batch_size, output_dim)
  """

  def __init__(self, vis_dim, hid_dim, with_batch=True, scale=0.08):
      """
      Initialize neural network.
      """
      self.vis_dim = vis_dim
      self.hid_dim = hid_dim
      self.with_batch = with_batch

      # Input gate weights
      self.w_xi = sharedX(rng.randn(vis_dim, hid_dim) * scale, name="LSTM.w_xi")
      self.w_hi = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="LSTM.w_hi")
      self.w_ci = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="LSTM.w_ci")

      # Forget gate weights
      #self.w_xf = sharedX(rng.randn(vis_dim, hid_dim) * scale, name="LSTM.w_xf")
      #self.w_hf = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="LSTM.w_hf")
      #self.w_cf = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="LSTM.w_cf")

      # Output gate weights
      self.w_xo = sharedX(rng.randn(vis_dim, hid_dim) * scale, name="LSTM.w_xo")
      self.w_ho = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="LSTM.w_ho")
      self.w_co = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="LSTM.w_co")

      # Cell weights
      self.w_xc = sharedX(rng.randn(vis_dim, hid_dim) * scale, name="LSTM.w_xc")
      self.w_hc = sharedX(rng.randn(hid_dim, hid_dim) * scale, name="LSTM.w_hc")

      # Initialize the bias vectors, c_0 and h_0 to zero vectors

      self.b_i = sharedX(numpy.zeros(hid_dim,), name="LSTM.b_i")
      self.b_f = sharedX(numpy.zeros(hid_dim,), name="LSTM.b_f")
      self.b_c = sharedX(numpy.zeros(hid_dim,), name="LSTM.b_c")
      self.b_o = sharedX(numpy.zeros(hid_dim,), name="LSTM.b_o")
      self.c_0 = sharedX(rng.randn(hid_dim,) * scale, name="LSTM.c_0")
      self.h_0 = sharedX(rng.randn(hid_dim,) * scale, name="LSTM.h_0")

      self.output_info = [self.c_0, self.h_0]
      # Define parameters
      self.params = [self.w_xi, self.w_hi, self.w_ci,
                     # self.w_xf, self.w_hf, self.w_cf,
                     self.w_xo, self.w_ho, self.w_co,
                     self.w_xc, self.w_hc,
                     self.b_i, self.b_c, self.b_o, # self.b_f, 
                     self.c_0, self.h_0]

 
  def fprop(self, x, bias=None):
      """
      Propagate the input through the network and return the last hidden vector.
      The whole sequence is also accessible through self.h
      """

      def recurrence(x_t, c_tm1, h_tm1):
          i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) + T.dot(h_tm1, self.w_hi) + T.dot(c_tm1, self.w_ci) + self.b_i)
          c_t = (1 - i_t) * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) + T.dot(h_tm1, self.w_hc) + self.b_c)
          o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) + T.dot(h_tm1, self.w_ho) + T.dot(c_t, self.w_co) + self.b_o)
          h_t = o_t * T.tanh(c_t)
          return c_t, h_t

      [_, h], _ = theano.scan(
          fn=recurrence,
          sequences=x,
          outputs_info=self.output_info,
          n_steps=x.shape[0]
      )
      return h
  
  
class DropoutLayer(object):
  """
  Dropout layer. Randomly set to 0 values of the input,
  with probability 1 - p
  """

  def __init__(self, p=0.5):
    """
    p has to be between 0 and 1.
    p is the probability of NOT dropping out a unit, so
    setting p to 1.0 is equivalent to have an identity layer.
    """
    assert 0. <= p <= 1.
    self.p = p
    self.rng = T.shared_randomstreams.RandomStreams(seed=123456)
    self.params = []
    
  def fprop(self, input, bias=None):
      """
      Dropout link: we just apply mask to the input.
      """
      mask = self.rng.binomial(n=1, p=self.p, size=input.shape, dtype=floatX)
      self.output = input * mask
      return self.output
            
if __name__ == "__main__":
  print "This is a library!"
  
  
  
  
