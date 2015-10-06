#!/usr/bin/env python

import argparse
import codecs
import os
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import mnlm 
import symbol_table as st

rng = numpy.random.RandomState(2016)

parser = argparse.ArgumentParser()
parser.add_argument('--lang_list', default="en_ru_fr_ro")
parser.add_argument('--corpus_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-corpus.")
parser.add_argument('--lang_vector_path', default="/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.")
parser.add_argument('--vector_size', type=int, default=70)
parser.add_argument('--ngram_order', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work")
parser.add_argument('--out_vectors', default="vectors")
parser.add_argument('--out_softmax_vectors', default="softmax_vectors")
parser.add_argument('--load_network', action='store_true', default=False)
parser.add_argument('--save_network', action='store_true', default=False)
parser.add_argument('--symbol_table', default="symbol_table")
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
  symbol_table_path = os.path.join(args.network_dir, args.lang_list, args.symbol_table)
  if os.path.exists(symbol_table_path):
    symbol_table.LoadFromFile(symbol_table_path)
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
  if args.load_network:
    network.LoadModel(args.network_dir)
  print "Training"
  prev_train_ppl = 100
  prev_dev_ppl = 100
  try:
    for epoch in xrange(args.num_epochs):
      train_logp, train_ppl = network.TrainEpoch(train_x, train_y, train_lang_feat, args.batch_size, lr=0.01)
      print "Epoch:", epoch+1
      print "Train cost mean:", train_logp, "perplexity:", train_ppl
      dev_logp, dev_ppl = network.Test(dev_x, dev_y, dev_lang_feat)
      print "Dev cost mean:", dev_logp, "perplexity:", dev_ppl
      if (dev_ppl - prev_dev_ppl) > 0.1 or abs(dev_ppl - train_ppl) < 0.001:
        # stop training if dev perplexity is growing or when train ppl equals dev ppl
        break
  except KeyboardInterrupt:
    print "Aborted. Saving to file."
  if args.save_network:
    network.SaveModel(args.network_dir)
  if args.out_vectors:
    out_vectors_path = os.path.join(args.network_dir, args.lang_list, args.out_vectors)
    SaveVectors(symbol_table, network.vectors, out_vectors_path)
  if args.out_softmax_vectors:
    out_softmax_vectors_path = os.path.join(args.network_dir, args.lang_list, args.out_softmax_vectors)
    softmax_vectors = network.SoftmaxVectors(train_x, train_y, train_lang_feat)
    SaveVectors(symbol_table, softmax_vectors, out_softmax_vectors_path)
  if args.symbol_table:
    symbol_table.SaveToFile(symbol_table_path)

if __name__ == '__main__':
    main()
