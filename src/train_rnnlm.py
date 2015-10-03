#!/usr/bin/env python

import argparse
import codecs
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

import rnnlm

parser = argparse.ArgumentParser()
parser.add_argument('--corpus')
parser.add_argument('--vectors', required=True)
parser.add_argument('--rnn_config')#, required=True
parser.add_argument('--ngram_order', type=int, default=3)
args = parser.parse_args()

def LoadVectors(filename):
  print "Loading vectors..."
  vectors = {}
  for line in codecs.open(filename, "r", "utf-8"):
    tokens = line.split()
    if len(tokens) < 3:
      continue
    word = tokens[0]
    if word == "</s>":
      word = '0'
    vectors[word] = [float(f) for f in tokens[1:]]
  print "done"
  return vectors
        
def flatten(l):
  return [item for sublist in l for item in sublist]

def LoadData(corpus, vectors, ngram_order):
  # in format: text corpus, embeddings, n-gram order
  # out format: for each n-gram, x: n-1 embeddings appended; y: n's word 1-hot representation
  x = []
  y = []
  len_one_hot = len(vectors.keys())
  word_ind = {}
  for i, word in enumerate(sorted(vectors.keys())):
    word_ind[word] = i
  def one_hot(word):
    return word_ind[word]
  def next_ngram(corpus):
    for line in codecs.open(corpus, "r", "utf-8"):
      tokens = line.split()
      #pad with 0 tokens
      for i in range(ngram_order-1):
        tokens = ['0'] + tokens + ['0']
      for ngram in zip(*[tokens[i:] for i in range(ngram_order)]):
        yield ngram
  def ngram_to_vectors(ngram):
    result = []
    for word in ngram:
      result.append(vectors[word])
    return flatten(result)
       
  for ngram in next_ngram(corpus):
    x.append(ngram_to_vectors(ngram[:-1]))
    y.append(one_hot(ngram[-1]))
      
  return numpy.array(x).astype("float32"), numpy.array(y).astype("float32"), len(word_ind)
  
def main(): 
  vectors = LoadVectors(args.vectors)
  x, y, vocab_size = LoadData(args.corpus, vectors, args.ngram_order)
  #samples = open("samples", "w")
  #labels = open("labels", "w") 
  #numpy.savetxt(samples, x, newline="\n")
  #numpy.savetxt(labels, y, newline="\n")
  
  train_x, dev_x, train_y, dev_y = train_test_split(x, y, test_size=0.2, random_state=42)

  hidden_dim = 20
  batch_size = 100
  epochs = 50  
 
  rnnlm.TrainRNNLM(train_x, train_y, vocab_size, hidden_dim, batch_size, epochs)

if __name__ == '__main__':
    main()
    
    
    
