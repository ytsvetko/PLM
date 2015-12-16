"""
RNNLM

"""
from __future__ import division

import os, sys
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

def flatten(l):
  return numpy.array([item for sublist in l for item in sublist])  

class MNLM(object):
  def __init__(self, vocab_size, vector_size, input_size, all_lang_symbol_indexes):
    # key = language symbol index, value = index in the all_lang_symbol_indexes list.
    self.lang_index_dict = {lang_ind: i for i, lang_ind in enumerate(all_lang_symbol_indexes)}
    self.vectors = numpy.zeros([vocab_size, vector_size])  # Current vectors

    print "vocab_size:", vocab_size
    print "vector_size:", vector_size
    print "input_size:", input_size

    self.x = T.imatrix("x")
    self.t = T.ivector("t")
    self.lang_indexes = T.ivector("lang_indexes")
    self.lang_vec = T.fmatrix("lang_vec")

    self.layers = [
          layer.Embeddings(vocab_size, vector_size, scale=0.08),
          layer.LSTM(input_size*vector_size, vector_size),                
          layer.DropoutLayer(),
          layer.Linear(vector_size, vocab_size), 
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

    self.test = theano.function(inputs=[self.x, self.t, self.lang_indexes, self.lang_vec], outputs=self.cost_fn,
                                allow_input_downcast=True, on_unused_input='ignore')

    self.loanword_log_prob = theano.function(inputs=[self.x, self.t, self.lang_indexes, self.lang_vec],
                                             outputs=self.log_prob_of_labels,
                                             allow_input_downcast=True, on_unused_input='ignore')

  def TrainEpoch(self, train_x, train_y, train_lang_indexes, lang_vec, batch_size, lr=0.01, to_shuffle=True):
    needs_flatten = type(train_x) == list
    assert len(train_x) == len(train_y) and len(train_y) == len(train_lang_indexes)
    assert len(lang_vec) == len(train_y)
      
    # Shuffle samples
    if to_shuffle:
      train_x, train_y, train_lang_indexes, lang_vec = shuffle(train_x, train_y, train_lang_indexes, lang_vec)
    assert len(train_x) == len(train_y) and len(train_y) == len(train_lang_indexes)
    assert len(lang_vec) == len(train_y)

    if needs_flatten:
      print len(train_x)
      train_x = flatten(list(train_x))
      print len(train_y)
      train_y = flatten(list(train_y))
      train_lang_indexes = flatten(list(train_lang_indexes))
      lang_vec = flatten(list(lang_vec))

    train_lang_indexes = self.ConvertLangIndexes(train_lang_indexes)
    
    ## Define update graph
    updates = learning_method.adam(self.cost_fn, self.params) 

    ## Compile Function
    train = theano.function(inputs=[self.x, self.t, self.lang_indexes, self.lang_vec],
                            outputs=self.cost_fn, updates=updates, 
                            allow_input_downcast=True, on_unused_input='ignore')
    nbatches = int(math.ceil(train_x.shape[0]/batch_size))

    train_costs = []
    for i in range(nbatches):
      start = i * batch_size
      end = min(start + batch_size, train_x.shape[0])
      cost = train(train_x[start:end], train_y[start:end], train_lang_indexes[start:end], lang_vec[start:end])
      train_costs.append(cost)
      self.UpdateVectors_()
      if i % 1000 == 0:
        print "Batch {}, cost: {}".format(i, cost)
    train_logp = numpy.mean(train_costs)
    train_ppl = numpy.power(2.0, train_logp)
    return train_logp, train_ppl

  def Test(self, x, y, lang_indexes, lang_vec, batch_size=100):
    if type(x) == list:
      x = flatten(x)
      y = flatten(y)
      lang_indexes = flatten(lang_indexes)
      lang_vec = flatten(lang_vec)
    lang_indexes = self.ConvertLangIndexes(lang_indexes)
    nbatches = int(math.ceil(x.shape[0]/batch_size))
    test_costs = []
    for i in range(nbatches):
      start = i * batch_size
      end = min(start + batch_size, x.shape[0])
      cost = self.test(x[start:end], y[start:end], lang_indexes[start:end], lang_vec[start:end])
      test_costs.append(cost)
      if i % 1000 == 0:
        print "Eval batch {}, cost: {}".format(i, cost)
    test_logp = numpy.mean(test_costs)
    test_ppl = numpy.power(2.0, test_logp)
    return test_logp, test_ppl
 
  def LogProb(self, x, y, lang_indexes, lang_vec, batch_size=100):
    lang_indexes = self.ConvertLangIndexes(lang_indexes)
    nbatches = int(math.ceil(x.shape[0]/batch_size))
    log_probs = []
    for i in range(nbatches):
      start = i * batch_size
      end = min(start + batch_size, x.shape[0])
      log_prob_vector = self.loanword_log_prob(x[start:end], y[start:end], lang_indexes[start:end], lang_vec[start:end])
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
