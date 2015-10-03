"""
Multimodal Neural Language Models

"""
from __future__ import division

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.utils import shuffle
import collections

import utils
import layer
import learning_method

rng = numpy.random.RandomState(2016)

def prop(layers, x, bias):
  for i, layer in enumerate(layers):
    if i == 0:
      layer_out = layer.fprop(x, bias)
    else:
      layer_out = layer.fprop(layer_out, bias)
  return layer_out

def get_params(layers, load_network_dump_path=None):
  params = []
  for layer in layers:
    params += layer.params
  if load_network_dump_path:
    utils.Load(layers, load_network_dump_path)
    UpdateVectors(layers[2].W.get_value())
  return params

def UpdateVectors(R):
  # R-- weight matrix from embedding layer
  # update VECTORS with learned weights
  assert len(VECTORS.keys()) == R.shape[0], (len(VECTORS.keys()), R.shape[0])
  for row, word in enumerate(sorted(VECTORS.keys())):
    VECTORS[word] = R[row]

def NgramToVector(train_x):
  def flatten(l):
    return [item for sublist in l for item in sublist]
  def ngram_to_vectors(ngram):
    result = []
    for word in ngram:
      result.append(VECTORS[word])
    return flatten(result)
  vec_train_x = []
  for ngram in train_x:
    vec_train_x.append(ngram_to_vectors(ngram))
  return numpy.array(vec_train_x).astype("float32")

class VectorMean(object):
  def __init__(self, size):
    self.vector = numpy.zeros(size)
    self.count = 0

  def Update(self, vector):
    self.vector += vector
    self.count += 1

  def Mean(self):
    return self.vector / self.count
    

def CollectSoftmaxVectors(prob_matrix, train_y):
  def init_val():
    return VectorMean(prob_matrix.shape[1])

  softmax_sums = collections.defaultdict(init_val)  # key: phone index, value: VectorMean 
  for i, phone_index in enumerate(train_y):
    softmax_sums[phone_index].Update(prob_matrix[i])

  phone_softmax = {}
  for i, phone in enumerate(sorted(VECTORS.keys())):
    phone_softmax[phone] = softmax_sums[i].Mean()
  return phone_softmax

class MNLM(object):
  def __init__(self, vectors, vector_size, context_size, ling_feat_size, out_dim):
    self.vectors = vectors  # Key: word; Value: current vectors
    self.layers = [
                layer.LBL_Biased(context_size*vector_size, vector_size, ling_feat_size, 
                                 ling_activation=T.nnet.sigmoid),
		layer.Activation(T.tanh),
                layer.Linear(vector_size, out_dim, scale=0.08), 
                layer.Activation(T.nnet.softmax),
            ]
   self.x, self.lang_bias, self.t = T.matrix("x"), T.matrix("lang_bias"), T.lvector("t")
   self.prob_fn = self.prop(layers, x, lang_bias)
   self.log_prob_fn = T.log(self.prob_fn)
   self.cost_fn = - T.mean(self.log_prob_fn[T.arange(self.t.shape[0]), self.t])

## Collect Parameters
params = get_params(layers, load_network_dump_path)


def TrainMNLM(train_x, dev_x, train_y, dev_y, train_ling_feat, dev_ling_feat, 
              vectors, vector_size, context_size, out_dim, batch_size=1, epochs=50,
              load_network_dump_path=None):
              
    x, lang_bias, t = T.matrix("x"), T.matrix("lang_bias"), T.lvector("t")
    global VECTORS
    VECTORS = vectors
    print("train_x.shape", train_x.shape)
    print("train_y.shape", train_y.shape)
    print("ling_feat.shape", train_ling_feat.shape)
    print("vocab size", out_dim)
    layers = [
                layer.LBL_Biased(context_size*vector_size, vector_size, train_ling_feat.shape[1], 
                                 ling_activation=T.nnet.sigmoid),
		layer.Activation(T.tanh),
                layer.Linear(vector_size, out_dim, scale=0.08), 
                layer.Activation(T.nnet.softmax),
            ]

    prob_fn = prop(layers, x, lang_bias)
    log_prob_fn = T.log(prob_fn)
    cost_fn = - T.mean(log_prob_fn[T.arange(t.shape[0]), t])

    ## Collect Parameters
    params = get_params(layers, load_network_dump_path)

    ## Define update graph
    updates = learning_method.sgd(cost_fn, params, lr=0.01) 

    ## Compile Function
    train = theano.function(inputs=[x, t, lang_bias], outputs=cost_fn, updates=updates, 
                            allow_input_downcast=True)
    test = theano.function(inputs=[x, t, lang_bias], outputs=cost_fn)

    softmax_probs = theano.function(inputs=[x, lang_bias], outputs=prob_fn)

    ## Train
    nbatches = train_x.shape[0]//batch_size
    for epoch in range(epochs):
      train_x, train_y, train_ling_feat = shuffle(train_x, train_y, train_ling_feat)  # Shuffle samples
      train_costs = []
      for i in range(nbatches):
        start = i * batch_size
        end = start + batch_size
        vec_train_x = NgramToVector(train_x[start:end]) 
        cost = train(vec_train_x, train_y[start:end], train_ling_feat[start:end])
        train_costs.append(cost)
        UpdateVectors(layers[2].W.get_value())
        if i % 1000 == 0:
          print "EPOCH:: %i, Iteration %i, cost: %.3f"%(epoch+1, i, cost)
      train_logp = numpy.mean(train_costs)
      train_ppl = numpy.power(2.0, train_logp)
      print "train cost mean:", train_logp, "train ppl:", train_ppl
      test_cost = test(NgramToVector(dev_x), dev_y, dev_ling_feat)
      print "dev cost:", test_cost, "dev ppl:", numpy.power(2.0, test_cost)

    prob_matrix = softmax_probs(NgramToVector(train_x), train_ling_feat)
    softmax_vectors = CollectSoftmaxVectors(prob_matrix, train_y)
    return VECTORS, layers, softmax_vectors

if __name__ == "__main__":
  print "This is a library!"
