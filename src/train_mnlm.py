#!/usr/bin/env python

import argparse
import codecs
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import random 
import os

import mnlm 
import symbol_table as st

random.seed(2016)

lang_list = "en_ru_fr_ro"
parser = argparse.ArgumentParser()
parser.add_argument('--lang_list', default="en_ru_fr_ro")
parser.add_argument('--corpus_path', default="/usr0/home/ytsvetko/projects/pnn/data/pron/pron-corpus.")
parser.add_argument('--lang_vector_path', default="/usr0/home/ytsvetko/projects/pnn/data/wals/feat.")
parser.add_argument('--vector_size', type=int, default=70)
parser.add_argument('--ngram_order', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--out_vectors', default="/usr0/home/ytsvetko/projects/pnn/work/"+lang_list+"/vectors")
parser.add_argument('--out_softmax_vectors', default="/usr0/home/ytsvetko/projects/pnn/work/"+lang_list+"/softmax_vectors")
parser.add_argument('--out_network', default="/usr0/home/ytsvetko/projects/pnn/work/"+lang_list)
parser.add_argument('--in_network', default="")
parser.add_argument('--symbol_table', default="/usr0/home/ytsvetko/projects/pnn/work/"+lang_list+"/symbol_table")


args = parser.parse_args()

start_symbol = "<s>"
end_symbol = "</s>"


def LoadData(corpus, symbol_table, ngram_order):
  # in format: text corpus, embeddings, n-gram order
  # out format: for each n-gram, x: n-1 embeddings appended; y: n's word 1-hot representation
  x = []
  y = []
  def next_ngram(corpus):
    for line in codecs.open(corpus, "r", "utf-8"):
      tokens = line.split()
      #pad with boundary tokens
      tokens = [start_symbol]*(ngram_order-1) + tokens + [end_symbol]
      for ngram in zip(*[tokens[i:] for i in range(ngram_order)]):
        yield ngram
       
  for ngram in next_ngram(corpus):
    x.append([symbol_table.WordIndex(word) for word in ngram[:-1]])
    y.append(symbol_table.WordIndex(ngram[-1]))
      
  return x, y

def LoadLangFeatVector(filename, num_samples):
  x = codecs.open(filename, "r", "utf-8").readlines()
  x = numpy.array([x]).astype("float32")
  return numpy.repeat(x, num_samples, 0)

def SaveVectors(symbol_table, vector_matrix, filename):
  out_f = codecs.open(filename, "w", "utf-8")
  for i, vector in enumerate(vector_matrix):
    vector = [str(num) for num in vector]
    out_f.write(u"{} {}\n".format(symbol_table.IndexToWord(i), " ".join(vector)))

def main():
  symbol_table = st.SymbolTable()
  if os.path.exists(args.symbol_table):
    symbol_table.LoadFromFile(args.symbol_table)
  x, y, lang_feat, vocab_size = None, None, None, 0
  for lang in args.lang_list.split("_"):
    print "Language:", lang
    corpus = args.corpus_path + lang
    lang_feat_vector = args.lang_vector_path + lang

    x_lang, y_lang = LoadData(corpus, symbol_table, args.ngram_order)
    lang_feat_lang = LoadLangFeatVector(lang_feat_vector, len(x_lang))

    if x is None: 
      x, y, lang_feat = numpy.array(x_lang), numpy.array(y_lang), numpy.array(lang_feat_lang)
    else:
      x = numpy.concatenate((x, x_lang), axis=0)
      y = numpy.concatenate((y, y_lang), axis=0)
      lang_feat = numpy.concatenate((lang_feat, lang_feat_lang), axis=0)

  train_x, dev_x, train_y, dev_y, train_lang_feat, dev_lang_feat = train_test_split(
      x, y, lang_feat, test_size=0.2, random_state=2016) 

  network = mnlm.MNLM(symbol_table.Size(), args.vector_size, args.ngram_order-1,
                      lang_feat.shape[1])
  if args.in_network:
    network.LoadModel(args.in_network)
  print "Training"
  try:
    for epoch in xrange(args.num_epochs):
      train_logp, train_ppl = network.TrainEpoch(train_x, train_y, train_lang_feat, args.batch_size, lr=0.01)
      print "Epoch:", epoch+1
      print "Train cost mean:", train_logp, "perplexity:", train_ppl
      dev_logp, dev_ppl = network.Test(dev_x, dev_y, dev_lang_feat)
      print "Dev cost mean:", dev_logp, "perplexity:", dev_ppl
  except KeyboardInterrupt:
    print "Aborted. Saving to file."
  if args.out_network:
    network.SaveModel(args.out_network)
  if args.out_vectors:
    SaveVectors(symbol_table, network.vectors, args.out_vectors)
  if args.out_softmax_vectors:
    softmax_vectors = network.SoftmaxVectors(train_x, train_y, train_lang_feat)
    SaveVectors(symbol_table, softmax_vectors, args.out_softmax_vectors)
  if args.symbol_table:
    symbol_table.SaveToFile(args.symbol_table)

if __name__ == '__main__':
    main()
