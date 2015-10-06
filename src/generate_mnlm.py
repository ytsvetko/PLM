#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import os

import mnlm 
import symbol_table as st

parser = argparse.ArgumentParser()
parser.add_argument('--lang_vector_path', default="/usr0/home/ytsvetko/projects/pnn/data/wals/feat.")
parser.add_argument('--vector_size', type=int, default=70)
parser.add_argument('--ngram_order', type=int, default=4)
parser.add_argument('--in_network', default="/usr0/home/ytsvetko/projects/pnn/work/en_ru")
parser.add_argument('--symbol_table', default="/usr0/home/ytsvetko/projects/pnn/work/en_ru/symbol_table")

args = parser.parse_args()

start_symbol = "<s>"
end_symbol = "</s>"

def LoadLangFeatVector(filename, num_samples):
  x = codecs.open(filename, "r", "utf-8").readlines()
  x = numpy.array([x]).astype("float32")
  return numpy.repeat(x, num_samples, 0)

def Generate(ngram_prefix, lang_feat, network, symbol_table, context_size, max_generated_len):
  str_ngram = [start_symbol]*context_size + ngram_prefix
  int_ngram = [symbol_table.WordIndex(w) for w in str_ngram]
  for iter_num in xrange(max_generated_len):
    int_next_symbol = network.PredictStochastic(int_ngram[-context_size:], lang_feat)
    int_ngram.append(int_next_symbol)
    str_next_symbol = symbol_table.IndexToWord(int_next_symbol)
    str_ngram.append(str_next_symbol)
    if str_next_symbol == end_symbol:
      break
  return str_ngram[context_size:-1]
  
def main():
  lang = "en"
  word = u't e l'
  symbol_table = st.SymbolTable()
  symbol_table.LoadFromFile(args.symbol_table)
  lang_feat_vector = args.lang_vector_path + lang
  lang_feat = LoadLangFeatVector(lang_feat_vector, 1)

  network = mnlm.MNLM(symbol_table.Size(), args.vector_size, args.ngram_order-1,
                      lang_feat.shape[1])
  network.LoadModel(args.in_network)
  generated_str = Generate(word.split(), lang_feat, network, 
                           symbol_table, args.ngram_order-1,
                           max_generated_len=30)
  print u" ".join(generated_str)

if __name__ == '__main__':
    main()
