#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import argparse
import codecs
import os
import sys
import numpy
import collections
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import marisa_trie

import mnlm 
import symbol_table as st
import levenshtein

parser = argparse.ArgumentParser()
parser.add_argument('--lang', default="fr_")
parser.add_argument('--pron_dict_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.")
parser.add_argument('--out_word_vectors_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.vectors.")
parser.add_argument('--lang_vector_path', default="/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.")
parser.add_argument('--vector_size', type=int, default=70)
parser.add_argument('--ngram_order', type=int, default=5)

parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/en_ru_fr_ro")
parser.add_argument('--network_epoch_dir', default="94")
parser.add_argument('--vectors', default="vectors")
parser.add_argument('--softmax_vectors', default="softmax_vectors")
parser.add_argument('--symbol_table', default="symbol_table")

parser.add_argument('--stochastic_sampling', action='store_true', default=False)
parser.add_argument('--constrain_prefixes', action='store_true', default=False)

args = parser.parse_args()

start_symbol = "<s>"
end_symbol = "</s>"

def LoadLangFeatVector(filename, num_samples):
  x = codecs.open(filename, "r", "utf-8").readlines()
  x = numpy.array([x]).astype("float32")
  return numpy.repeat(x, num_samples, 0)


def LoadCorpus(corpus_path):
  trie = marisa_trie.Trie()
  corpus = set()
  for line in codecs.open(corpus_path, "r", "utf-8"):
    tokens = line.strip().split(" ||| ")
    if len(tokens) != 2: 
      continue
    word, pronunciation = tokens
    corpus.add(pronunciation)
  return marisa_trie.Trie(corpus)
  
def Generate(ngram_prefix, lang_feat, network, symbol_table, 
             context_size, max_generated_len, corpus, 
             stochastic_sampling=False, constrain_prefixes=False):
  str_ngram = [start_symbol]*context_size + ngram_prefix
  int_ngram = [symbol_table.WordIndex(w) for w in str_ngram]
  str_prefix = u" ".join(ngram_prefix)
  next_max_in_softmax = 1
  
  if constrain_prefixes and not corpus.items(str_prefix):
    return None
    
  for iter_num in xrange(max_generated_len):
    if stochastic_sampling:
      int_next_symbol = network.PredictStochastic(int_ngram[-context_size:], lang_feat)
    else:
      int_next_symbol = network.Predict(int_ngram[-context_size:], lang_feat, next_max_in_softmax)
    str_next_symbol = symbol_table.IndexToWord(int_next_symbol)
    if str_next_symbol == end_symbol:
      if constrain_prefixes and not(str_prefix in corpus):
        next_max_in_softmax += 1
        continue
      else:
        break
    if constrain_prefixes and not corpus.items(str_prefix + u" " + str_next_symbol):
      # Generated prefix not in the pronunciation dictionary; try again
      next_max_in_softmax += 1
      continue 
    next_max_in_softmax = 1
    str_prefix += u" " + str_next_symbol
    print str_prefix
    str_ngram.append(str_next_symbol)
    int_ngram.append(int_next_symbol)
  return str_ngram[context_size:]

def WordVectorContextSoftmax(word, lang_feat, network, symbol_table, context_size):
  str_ngram = [start_symbol]*context_size + word
  int_ngram = [symbol_table.WordIndex(w) for w in str_ngram]
  
  ngram_vectors = []
  for ngram in zip(*[int_ngram[i:] for i in range(context_size)]):
    ngram_vectors.append(network.ProbVectorGivenPrefix(ngram, lang_feat).tolist())
  assert len(ngram_vectors[:-1]) == len(word), (len(ngram_vectors), len(word))
  return ngram_vectors[:-1] #numpy.mean(ngram_vectors, axis=0)
   
def WordVectorSoftmax(word, lang_feat, network, symbol_table):
  vectors = []
  for w in word:
    int_w = symbol_table.WordIndex(w)
    vectors.append(network.softmax_vectors[int_w].tolist())
  assert len(vectors) == len(word), (len(vectors), len(word))
  return vectors 

def WordVectorR(word, network, symbol_table):
  vectors = []
  for w in word:
    int_w = symbol_table.WordIndex(w)
    vectors.append(network.vectors[int_w].tolist())
  assert len(vectors) == len(word), (len(vectors), len(word))
  return vectors 
   
def GenerateWordVectorsForLanguage(in_filename, out_filename,  
               lang_feat, network, symbol_table, context_size):
  def str_vector(word_vectors):
    result = []
    for word_vector in word_vectors:
      result.append(" ".join(["%.8f" % number for number in word_vector]))
    return "\t".join(result)
    
  out_f = codecs.open(out_filename, "w", "utf-8")  
  for line_num, line in enumerate(codecs.open(in_filename, "r", "utf-8")):
    tokens = line.strip().split(" ||| ")
    if len(tokens) != 2: 
      continue
    if line_num % 1000 == 0:
      print line_num
    word, pron = tokens
    word_vector_softmax = WordVectorSoftmax(pron.split(), lang_feat, 
          network, symbol_table)
    word_vector_context_softmax = WordVectorContextSoftmax(pron.split(), lang_feat, 
          network, symbol_table, context_size)      
    word_vector_r = WordVectorR(pron.split(), network, symbol_table)
        
    #word_vector = word_vectors[word][0]
    out_f.write(u"{} ||| {} ||| {} ||| {} ||| {}\n".format(word, pron, 
          str_vector(word_vector_softmax), str_vector(word_vector_context_softmax), 
          str_vector(word_vector_r)))


def main():
  print "Language:", args.lang
  max_generated_len=sys.maxint
  corpus_path = args.pron_dict_path + args.lang
    
  lang_feat_vector = args.lang_vector_path + args.lang
  lang_feat = LoadLangFeatVector(lang_feat_vector, 1)
  
  symbol_table = st.SymbolTable()
  symbol_table_path = os.path.join(args.network_dir, args.symbol_table)
  symbol_table.LoadFromFile(symbol_table_path)
    
  network = mnlm.MNLM(symbol_table.Size(), args.vector_size, 
                      args.ngram_order-1, lang_feat.shape[1])
  network.LoadModel(os.path.join(args.network_dir, args.network_epoch_dir))
  
  GenerateWordVectorsForLanguage(corpus_path, args.out_word_vectors_path + args.lang, 
                   lang_feat, network, symbol_table, args.ngram_order-1)
                           
  """
  # Generate words
  prefix = u"ɑː g"
  corpus = LoadCorpus(corpus_path) # Pronunciation dictionary trie
  generated_str = Generate(prefix.split(), lang_feat, network, 
                           symbol_table, args.ngram_order-1,
                           max_generated_len, corpus,
                           args.stochastic_sampling, args.constrain_prefixes)
  if generated_str:
    print u" ".join(generated_str)
  """
  
if __name__ == '__main__':
    main()
