#!/usr/bin/env python

import argparse
import codecs
import os
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score

import mnlm 
import symbol_table as st

parser = argparse.ArgumentParser()
parser.add_argument('--lang_list', default="en_ru_fr_ro_it_mt")
parser.add_argument('--train_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/train/pron-corpus.")
parser.add_argument('--dev_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-corpus.")
parser.add_argument('--lang_vector_path', default="/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.")
parser.add_argument('--vector_size', type=int, default=70)
parser.add_argument('--ngram_order', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=100)
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

def AppendLangData(symbol_table, ngram_order, corpus_filename, lang_feat_vector_filename, x, y, lang_feat):
  x_lang, y_lang = LoadData(corpus_filename, symbol_table, ngram_order)
  lang_feat_lang = LoadLangFeatVector(lang_feat_vector_filename, len(x_lang))
  if x is None:
    return numpy.array(x_lang), numpy.array(y_lang), numpy.array(lang_feat_lang)
  else:
    x = numpy.concatenate((x, x_lang), axis=0)
    y = numpy.concatenate((y, y_lang), axis=0)
    lang_feat = numpy.concatenate((lang_feat, lang_feat_lang), axis=0)
    return x, y, lang_feat
  
def main():
  try:
    os.stat(os.path.join(args.network_dir, args.lang_list))
  except:
    os.mkdir(os.path.join(args.network_dir, args.lang_list))
    
  symbol_table = st.SymbolTable()
  symbol_table_path = os.path.join(args.network_dir, args.lang_list, args.symbol_table)
  if os.path.exists(symbol_table_path):
    symbol_table.LoadFromFile(symbol_table_path)
  train_x, train_y, train_lang_feat = None, None, None
  dev_x, dev_y, dev_lang_feat = None, None, None
  for lang in args.lang_list.split("_"):
    print "Language:", lang
    lang_feat_vector = args.lang_vector_path + lang
    train_x, train_y, train_lang_feat = AppendLangData(
        symbol_table, args.ngram_order, args.train_path + lang,
        lang_feat_vector, train_x, train_y, train_lang_feat)
    dev_x, dev_y, dev_lang_feat = AppendLangData(
        symbol_table, args.ngram_order, args.dev_path + lang,
        lang_feat_vector, dev_x, dev_y, dev_lang_feat)

  network = mnlm.MNLM(symbol_table.Size(), args.vector_size, args.ngram_order-1,
                      train_lang_feat.shape[1])
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
      #if (dev_ppl - prev_dev_ppl) > 0.1 or abs(dev_ppl - train_ppl) < 0.0001:
      #  # stop training if dev perplexity is growing or when train ppl equals dev ppl
      #  break
      #prev_dev_ppl = dev_ppl
      os.mkdir(os.path.join(args.network_dir, args.lang_list, str(epoch+1)))
      if args.save_network:
        network.SaveModel(os.path.join(args.network_dir, args.lang_list, str(epoch+1)))
      if args.out_vectors:
        out_vectors_path = os.path.join(args.network_dir, args.lang_list, str(epoch+1), args.out_vectors)
        SaveVectors(symbol_table, network.vectors, out_vectors_path)
 
  except KeyboardInterrupt:
    print "Aborted. Saving to file."
  #if args.out_softmax_vectors:
  #  out_softmax_vectors_path = os.path.join(args.network_dir, args.lang_list, args.out_softmax_vectors)
  #  softmax_vectors = network.SoftmaxVectors(train_x, train_y, train_lang_feat)
  #  SaveVectors(symbol_table, softmax_vectors, out_softmax_vectors_path)
  if args.symbol_table:
    symbol_table.SaveToFile(symbol_table_path)

if __name__ == '__main__':
    main()
