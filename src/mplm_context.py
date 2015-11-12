"""
Multimodal Neural Language Models

"""
from __future__ import division

import os
import cPickle
import numpy
import theano
import theano.tensor as T
from sklearn.utils import shuffle
import collections
import heapq
import random 
import math

import layer
import learning_method

class MNLM(object):
  def __init__(self, vocab_size, vector_size, input_size, all_lang_symbol_indexes):
    # key = language symbol index, value = index in the all_lang_symbol_indexes list.
    self.lang_index_dict = {lang_ind: i for i, lang_ind in enumerate(all_lang_symbol_indexes)}
    self.vectors = numpy.zeros([vocab_size, vector_size])  # Current vectors

    self.x = T.imatrix("x")
    self.lang_bias = T.matrix("lang_bias")
    self.t = T.ivector("t")
    self.lang_indexes = T.ivector("lang_indexes")

    embeddings_layer = layer.Embeddings(vocab_size, vector_size, scale=0.08)
    self.layers = [
          embeddings_layer,
          layer.Linear(input_size*vector_size, vector_size),
          layer.Activation(T.tanh),
          layer.Concatenation(self.lang_indexes, embeddings_layer),  # Must be after tanh layer.
          layer.Linear(2 * vector_size, vector_size, scale=0.08), 
          layer.Activation(T.tanh),
          layer.Linear(vector_size, vocab_size, scale=0.08), 
          layer.Activation(T.nnet.softmax),
        ]

    self.prob_fn = self.Propagate_()
    # self.log_prob_fn is a matrix of batch_size x vocab_size
    # Element (n, i) in it is a log probability of symbol 'i' given input ngram 'n'.
    self.log_prob_fn = T.log(self.prob_fn)
    
    # self.t is a vector of label indexes. Each element is in the range of (0 to vocab_size).
    # len(self.t) == batch size
    symbolic_batch_size = self.t.shape[0]
    # Get log prob of each gold label in the batch.
    self.log_prob_of_labels = self.log_prob_fn[T.arange(symbolic_batch_size), self.t]
    self.cost_fn = - T.mean(self.log_prob_of_labels)
    
    ## Collect Parameters
    self.params = self.GetParams_()

    self.test = theano.function(inputs=[self.x, self.t, self.lang_indexes], outputs=self.cost_fn,
                                allow_input_downcast=True, on_unused_input='ignore')

    self.loanword_log_prob = theano.function(inputs=[self.x, self.t, self.lang_indexes],
                                             outputs=self.log_prob_of_labels,
                                             allow_input_downcast=True, on_unused_input='ignore')

  def TrainEpoch(self, train_x, train_y, train_lang_indexes, batch_size, lr=0.01):
    train_lang_indexes = self.ConvertLangIndexes(train_lang_indexes)
    ## Define update graph
    updates = learning_method.adam(self.cost_fn, self.params) 

    ## Compile Function
    train = theano.function(inputs=[self.x, self.t, self.lang_indexes],
                            outputs=self.cost_fn, updates=updates, 
                            allow_input_downcast=True, on_unused_input='ignore')
    nbatches = int(math.ceil(train_x.shape[0]/batch_size))
    # Shuffle samples
    train_x, train_y, train_lang_indexes = shuffle(train_x, train_y, train_lang_indexes)
    train_costs = []
    for i in range(nbatches):
      start = i * batch_size
      end = min(start + batch_size, train_x.shape[0])
      cost = train(train_x[start:end], train_y[start:end], train_lang_indexes[start:end])
      train_costs.append(cost)
      self.UpdateVectors_()
      if i % 1000 == 0:
        print "Batch {}, cost: {}".format(i, cost)
    train_logp = numpy.mean(train_costs)
    train_ppl = numpy.power(2.0, train_logp)
    return train_logp, train_ppl

  def Test(self, x, y, lang_indexes, batch_size=100):
    lang_indexes = self.ConvertLangIndexes(lang_indexes)
    nbatches = int(math.ceil(x.shape[0]/batch_size))
    test_costs = []
    for i in range(nbatches):
      start = i * batch_size
      end = min(start + batch_size, x.shape[0])
      cost = self.test(x[start:end], y[start:end], lang_indexes[start:end])
      test_costs.append(cost)
      if i % 1000 == 0:
        print "Eval batch {}, cost: {}".format(i, cost)
    test_logp = numpy.mean(test_costs)
    test_ppl = numpy.power(2.0, test_logp)
    return test_logp, test_ppl
 
  def LogProb(self, x, y, lang_indexes, batch_size=100):
    lang_indexes = self.ConvertLangIndexes(lang_indexes)
    nbatches = int(math.ceil(x.shape[0]/batch_size))
    log_probs = []
    for i in range(nbatches):
      start = i * batch_size
      end = min(start + batch_size, x.shape[0])
      log_prob_vector = self.loanword_log_prob(x[start:end], y[start:end], lang_indexes[start:end])
      log_probs.extend(log_prob_vector)
      if i % 1000 == 0:
        print "Eval batch {}".format(i)
    return log_probs
    
  def Propagate_(self):
    layer_out = self.x
    for i, layer in enumerate(self.layers):
      print "Composing layer", i
      layer_out = layer.fprop(layer_out)
    return layer_out

  def GetParams_(self):
    params = []
    for layer in self.layers:
      params += layer.params
    return params
    
  def LoadModel(self, dirname): 
    # Load components
    for l, layer in enumerate(self.layers):
      for p, param in enumerate(layer.params):
        param_path = os.path.join(dirname, "{}_{}.pkl".format(l, p))
        param.set_value(cPickle.load(open(param_path, 'r')))
    self.UpdateVectors_()

  def SaveModel(self, dirname):
    # Write components
    for l, layer in enumerate(self.layers):
      for p, param in enumerate(layer.params):
        param_path = os.path.join(dirname, "{}_{}.pkl".format(l, p))
        cPickle.dump(param.get_value(), open(param_path, 'w'))
   
  def UpdateVectors_(self):
    # R-- weight matrix from embedding layer
    # update VECTORS with learned weights
    R = self.layers[0].M.get_value()
    assert self.vectors.shape == R.shape, (self.vectors.shape, R.shape)
    self.vectors = R

  def ConvertLangIndexes(self, lang_indexes):
    return numpy.array([self.lang_index_dict[i] for i in lang_indexes])

if __name__ == "__main__":
  print "This is a library!"
