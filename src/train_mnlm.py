#!/usr/bin/env python

import argparse
import codecs
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import random 

import mnlm 

random.seed(2016)

parser = argparse.ArgumentParser()
parser.add_argument('--lang_list', nargs='+', default="en")
parser.add_argument('--corpus_path', default="/usr0/home/ytsvetko/projects/pnn/data/pron/pron-corpus.")
parser.add_argument('--ling_vector_path', default="/usr0/home/ytsvetko/projects/pnn/data/wals/feat.")
parser.add_argument('--vectors', default="/usr0/home/ytsvetko/projects/pnn/data/corpus/vectors")
parser.add_argument('--ngram_order', type=int, default=4)
parser.add_argument('--out_vectors', default="/usr0/home/ytsvetko/projects/pnn/work/vectors.en")
parser.add_argument('--out_softmax_vectors', default="/usr0/home/ytsvetko/projects/pnn/work/softmax_vectors.en")
parser.add_argument('--out_network', default="/usr0/home/ytsvetko/projects/pnn/work")
parser.add_argument('--in_network', default="/usr0/home/ytsvetko/projects/pnn/work")
args = parser.parse_args()

start_symbol = "<s>"
end_symbol = "</s>"

def LoadVectors(filename, init_random=False, dim=70):
  print "Loading vectors..."
  vectors = {}
  vector_size = 0
  for line in codecs.open(filename, "r", "utf-8"):
    tokens = line.split()
    if len(tokens) < 3:
      continue
    word = tokens[0]
    if word == "<s>":
      word = start_symbol    
    if word == "</s>":
      word = end_symbol
    if init_random: 
      vectors[word] = [random.uniform(-0.08, 0.08) for _ in range(dim)]
    else:
      vectors[word] = [float(f) for f in tokens[1:]]
    if vector_size == 0 and not init_random:
      vector_size = len(tokens[1:])
    else:
      vector_size = dim
  print "done"
  return vectors, vector_size

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
      #pad with boundary tokens
      tokens = [start_symbol]*(ngram_order-1) + tokens + [end_symbol]
      for ngram in zip(*[tokens[i:] for i in range(ngram_order)]):
        yield ngram
       
  for ngram in next_ngram(corpus):
    x.append(ngram[:-1])
    y.append(one_hot(ngram[-1]))
      
  return x, y, word_ind

def LoadLingFeatVector(filename, num_samples):
  x = codecs.open(filename, "r", "utf-8").readlines()
  x = numpy.array([x]).astype("float32")
  return numpy.repeat(x, num_samples, 0)

def SaveVectors(vectors, filename):
  out_f = codecs.open(filename, "w", "utf-8")
  for word, vector in sorted(vectors.items()):
    vector = [str(num) for num in vector]
    out_f.write(u"{} {}\n".format(word, " ".join(vector)))

def main(): 
  vectors, vector_size = LoadVectors(args.vectors, init_random=True)
  x, y, ling_feat, vocab_size = None, None, None, 0
  for lang in args.lang_list.split():
    print "Language:", lang
    corpus = args.corpus_path + lang
    ling_feat_vector = args.ling_vector_path + lang

    x_lang, y_lang, one_hot_mapping = LoadData(corpus, vectors, args.ngram_order)
    ling_feat_lang = LoadLingFeatVector(ling_feat_vector, len(x_lang))

    if x is None: 
      x, y, ling_feat = numpy.array(x_lang), numpy.array(y_lang), numpy.array(ling_feat_lang)
    else:
      x = numpy.concatenate((x, x_lang), axis=0)
      y = numpy.concatenate((y, y_lang), axis=0)
      ling_feat = numpy.concatenate((ling_feat, ling_feat_lang), axis=0)

  
  train_x, dev_x, train_y, dev_y, train_ling_feat, dev_ling_feat = train_test_split(
      x, y, ling_feat, test_size=0.2, random_state=2016) 
  batch_size = 50
  epochs =  50

  network = mnlm.MNLM(vectors, vector_size, args.ngram_order-1,
                      ling_feat.shape[1], len(one_hot_mapping.keys()))
  if args.in_network:
    network.LoadModel(args.in_network)
  print "Training"
  for epoch in xrange(epochs):
    train_logp, train_ppl = network.TrainEpoch(train_x, train_y, train_ling_feat, batch_size, lr=0.01)
    print "Epoch:", epoch+1
    print "Train cost mean:", train_logp, "perplexity:", train_ppl
    dev_logp, dev_ppl = network.Test(dev_x, dev_y, dev_ling_feat)
    print "Dev cost mean:", dev_logp, "perplexity:", dev_ppl
  if args.out_network:
    network.SaveModel(args.out_network)
  if args.out_vectors:
    SaveVectors(network.vectors, args.out_vectors)
  if args.out_softmax_vectors:
    softmax_vectors = network.SoftmaxVectors(train_x, train_y, train_ling_feat)
    SaveVectors(softmax_vectors, args.out_softmax_vectors)

if __name__ == '__main__':
    main()
