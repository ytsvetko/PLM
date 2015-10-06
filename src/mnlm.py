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
import random 

random.seed(1234)

import layer
import learning_method

class MNLM(object):
  def __init__(self, vocab_size, vector_size, context_size, lang_feat_size):
    self.vectors = numpy.zeros([vocab_size, vector_size])  # Current vectors
    self.layers = [
                layer.LBL_Biased(context_size*vector_size, vector_size, lang_feat_size, 
                                 lang_activation=T.nnet.sigmoid),
                layer.Activation(T.tanh),
                layer.Linear(vector_size, vocab_size, scale=0.08), 
                layer.Activation(T.nnet.softmax),
            ]

    self.x = T.matrix("x")
    self.lang_bias = T.matrix("lang_bias")
    self.t = T.lvector("t")

    self.prob_fn = self.Propagate_()
    self.log_prob_fn = T.log(self.prob_fn)
    self.cost_fn = - T.mean(self.log_prob_fn[T.arange(self.t.shape[0]), self.t])

    ## Collect Parameters
    self.params = self.GetParams_()

    self.softmax_probs = theano.function(inputs=[self.x, self.lang_bias], outputs=self.prob_fn)
    self.test = theano.function(inputs=[self.x, self.t, self.lang_bias], outputs=self.cost_fn)    

  def TrainEpoch(self, train_x, train_y, train_lang_feat, batch_size, lr=0.01):
    ## Define update graph
    updates = learning_method.sgd(self.cost_fn, self.params, lr=lr) 

    ## Compile Function
    train = theano.function(inputs=[self.x, self.t, self.lang_bias],
                            outputs=self.cost_fn, updates=updates, 
                            allow_input_downcast=True)

    if batch_size > train_x.shape[0]:
      batch_size = train_x.shape[0]
    nbatches = train_x.shape[0]//batch_size
    # Shuffle samples
    train_x, train_y, train_lang_feat = shuffle(train_x, train_y, train_lang_feat)
    train_costs = []
    for i in range(nbatches):
      start = i * batch_size
      end = start + batch_size
      vec_train_x = self.NgramToVector_(train_x[start:end]) 
      cost = train(vec_train_x, train_y[start:end], train_lang_feat[start:end])
      train_costs.append(cost)
      self.UpdateVectors_()
      if i % 1000 == 0:
        print "Batch {}, cost: {}".format(i, cost)
    train_logp = numpy.mean(train_costs)
    train_ppl = numpy.power(2.0, train_logp)
    return train_logp, train_ppl

  def Test(self, x, y, lang_feat):
    test_cost = self.test(self.NgramToVector_(x), y, lang_feat)
    test_ppl = numpy.power(2.0, test_cost)
    return test_cost, test_ppl

  def SoftmaxVectors(self, x, y, lang_feat):
    # Helper function for averaging softmax predictions for each value of 'y'
    def CollectSoftmaxVectors(prob_matrix):
      class VectorMean(object):
        def __init__(self):
          size = prob_matrix.shape[1]
          self.vector = numpy.zeros(size)
          self.count = 0
        def Update(self, vector):
          self.vector += vector
          self.count += 1
        def Mean(self):
          return self.vector / self.count

      # Sum up all vectors for each 'y'
      softmax_sums = collections.defaultdict(VectorMean)  # key: phone index, value: VectorMean 
      for i, phone_index in enumerate(y):
        softmax_sums[phone_index].Update(prob_matrix[i])

      # Calculate the average for each 'y' and convert y indexes into phones.
      phone_softmax = []
      for i in xrange(self.vectors.shape[0]):
        phone_softmax.append(softmax_sums[i].Mean())
      return phone_softmax

    prob_matrix = self.softmax_probs(self.NgramToVector_(x), lang_feat)
    return CollectSoftmaxVectors(prob_matrix)
  
  def Predict(self, ngram, lang_feat):
    prob_matrix = self.softmax_probs(self.NgramToVector_([ngram]), lang_feat)
    assert prob_matrix.shape[0] == 1, prob_matrix.shape
    return numpy.argmax(prob_matrix, axis=1)[0]
    
  def PredictStochastic(self, ngram, lang_feat):
    prob_matrix = self.softmax_probs(self.NgramToVector_([ngram]), lang_feat)
    assert prob_matrix.shape[0] == 1, prob_matrix.shape
    cumulative_prob = 0
    rand_ind = random.random()
    for ind, prob in enumerate(prob_matrix[0]):
      cumulative_prob += prob
      if rand_ind < cumulative_prob:
        return ind
    return numpy.argmax(prob_matrix, axis=1)[0]
    
  def NgramToVector_(self, train_x):
    def flatten(l):
      return [item for sublist in l for item in sublist]
    def ngram_to_vectors(ngram):
      result = []
      for word_ind in ngram:
        result.append(self.vectors[word_ind])
      return flatten(result)
    vec_train_x = []
    for ngram in train_x:
      vec_train_x.append(ngram_to_vectors(ngram))
    return numpy.array(vec_train_x).astype("float32")

  def Propagate_(self):
    for i, layer in enumerate(self.layers):
      if i == 0:
        layer_out = layer.fprop(self.x, self.lang_bias)
      else:
        layer_out = layer.fprop(layer_out, self.lang_bias)
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
    R = self.layers[2].W.get_value().transpose()
    assert self.vectors.shape == R.shape, (self.vectors.shape, R.shape)
    self.vectors = R

if __name__ == "__main__":
  print "This is a library!"
