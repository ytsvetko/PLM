#!/usr/bin/env python

import argparse
import codecs
import os
import numpy

import mnlm 
import symbol_table as st

parser = argparse.ArgumentParser()
parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/zero/en/100")
parser.add_argument('--lang_list', default="en")
parser.add_argument('--lang_vector_path', default="/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.")
parser.add_argument('--symbol_table', default="/usr1/home/ytsvetko/projects/mnlm/work/symbol_table.en_ru_fr_ro_it_mt")

parser.add_argument('--dev_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en")
parser.add_argument('--test_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en")

parser.add_argument('--vector_size', type=int, default=90)
parser.add_argument('--ngram_order', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=100)
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
    
  symbol_table = st.SymbolTable()
  if os.path.exists(args.symbol_table):
    symbol_table.LoadFromFile(args.symbol_table)
    
  dev_x, dev_y, dev_lang_feat = None, None, None
  test_x, test_y, test_lang_feat = None, None, None
  for lang in args.lang_list.split("_"):
    print "Language:", lang
    lang_feat_vector = args.lang_vector_path + lang
    if args.dev_path:
      dev_x, dev_y, dev_lang_feat = AppendLangData(
          symbol_table, args.ngram_order, args.dev_path,
          lang_feat_vector, dev_x, dev_y, dev_lang_feat)
    if args.test_path:
      test_x, test_y, test_lang_feat = AppendLangData(
          symbol_table, args.ngram_order, args.test_path,
          lang_feat_vector, test_x, test_y, test_lang_feat)

  network = mnlm.MNLM(symbol_table.Size(), args.vector_size, args.ngram_order-1,
                      dev_lang_feat.shape[1])
  
  network.LoadModel(args.network_dir)
  
  if args.dev_path:
    print "Dev set evaluation"
    dev_logp, dev_ppl = network.Test(dev_x, dev_y, dev_lang_feat)
    print "Dev cost mean:", dev_logp, "perplexity:", dev_ppl

  if args.test_path:
    print "Test set evaluation"
    test_logp, test_ppl = network.Test(test_x, test_y, test_lang_feat)
    print "Test cost mean:", test_logp, "perplexity:", test_ppl


if __name__ == '__main__':
    main()
