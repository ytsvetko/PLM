#!/usr/bin/env python

import argparse
import codecs
import os
import numpy

import mplm_factored as mplm
import symbol_table as st

parser = argparse.ArgumentParser()
parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/context")
parser.add_argument('--lang_list', default="ar")
parser.add_argument('--symbol_table', default="/usr1/home/ytsvetko/projects/mnlm/work/symbol_table.en_ru_fr_ro_it_mt_sw_hi_ar")
parser.add_argument('--lang_vec_file_prefix', default="/usr1/home/ytsvetko/projects/mnlm/data/typology/feat.")
parser.add_argument('--dev_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.ar")
parser.add_argument('--test_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.ar")
parser.add_argument('--vector_size', type=int, default=100)
parser.add_argument('--context_size', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--betta', type=float, default=0.0)

args = parser.parse_args()

start_symbol = "<s>"
end_symbol = "</s>"

insert_symbol = "INS"
delete_symbol = "DEL"


def LoadLangVec(lang_vec_file_prefix, lang_list):
  result = {}
  for lang in lang_list:
    vector = []
    for line in open(lang_vec_file_prefix + lang):
      vector.append(float(line.strip()))
    result[lang] = vector
  return result

def LoadData(corpus, symbol_table, context_size, lang, lang_vector):
  # in format: text corpus, embeddings, n-gram order
  # out format: for each n-gram, x: n-1 embeddings appended; y: n's word 1-hot representation
  lang_symbol = symbol_table.WordIndex(lang)
  def next_ngram(corpus):
    for line in codecs.open(corpus, "r", "utf-8"):
      new_word = True
      tokens = line.split()
      #pad with boundary tokens
      tokens = [start_symbol]*context_size + tokens + [end_symbol]
      for ngram in zip(*[tokens[i:] for i in range(1+context_size)]):
        yield ngram, new_word
        new_word = False
       
  x = []
  y = []
  lang_symbols = []
  lang_vectors = []
  w_x, w_y, w_lang, w_lang_vec = [], [], [], []
  for ngram, new_word in next_ngram(corpus):
    if new_word and len(w_x)>0: 
      x.append(w_x)
      y.append(w_y)
      lang_symbols.append(w_lang)
      lang_vectors.append(w_lang_vec)
      w_x, w_y, w_lang, w_lang_vec = [], [], [], []
      
    w_x.append([symbol_table.WordIndex(word) for word in ngram[:-1]] + 
             [lang_symbol])
    w_y.append(symbol_table.WordIndex(ngram[-1]))
    w_lang.append(lang_symbol)
    w_lang_vec.append(lang_vector)

  if len(w_x)>0: 
   x.append(w_x)
   y.append(w_y)
   lang_symbols.append(w_lang)
   lang_vectors.append(w_lang_vec)
  return x, y, lang_symbols, lang_vectors

def SaveVectors(symbol_table, vector_matrix, filename):
  out_f = codecs.open(filename, "w", "utf-8")
  for i, vector in enumerate(vector_matrix):
    vector = [str(num) for num in vector]
    out_f.write(u"{} {}\n".format(symbol_table.IndexToWord(i), " ".join(vector)))

def AppendLangData(symbol_table, context_size, corpus_filename, lang_vector,
                   x, y, lang, lang_symbols, lang_vectors):
  x_lang, y_lang, lang_symbols_lang, lang_vectors_lang = LoadData(corpus_filename, symbol_table, context_size, lang, lang_vector)
  if not x:
    x = x_lang
    y = y_lang
    lang_symbols = lang_symbols_lang
    lang_vectors = lang_vectors_lang
  else:
    x.extend(x_lang)
    y.extend(y_lang)
    lang_symbols.extend(lang_symbols_lang)
    lang_vectors.extend(lang_vectors_lang)
  return x, y, lang_symbols, lang_vectors

  
def GetLanguageList(languages, symbol_table):
  return numpy.array([symbol_table.WordIndex(l) for l in languages])

def main():
    
  symbol_table = st.SymbolTable()
  if os.path.exists(args.symbol_table):
    symbol_table.LoadFromFile(args.symbol_table)

  lang_list = args.lang_list.split("_")
  lang_vec_dict = LoadLangVec(args.lang_vec_file_prefix, lang_list)

  dev_x, dev_y, dev_lang_feat, dev_lang_vec = None, None, None, None
  test_x, test_y, test_lang_feat, test_lang_vec = None, None, None, None
  for lang in lang_list:
    print "Language:", lang
    if args.dev_path:
      dev_x, dev_y, dev_lang_feat, dev_lang_vec = AppendLangData(
          symbol_table, args.context_size, args.dev_path, lang_vec_dict[lang],
          dev_x, dev_y, lang, dev_lang_feat, dev_lang_vec)
    if args.test_path:
      test_x, test_y, test_lang_feat, test_lang_vec = AppendLangData(
          symbol_table, args.context_size, args.test_path, lang_vec_dict[lang],
          test_x, test_y, lang, test_lang_feat, test_lang_vec)

  all_lang_symbol_indexes = GetLanguageList(args.symbol_table.split(".")[-1].split("_"), symbol_table)
  network = mplm.MNLM(symbol_table.Size(), args.vector_size, 1+args.context_size, all_lang_symbol_indexes)
  
  network.LoadModel(args.network_dir)
  
  if args.dev_path:
    print "Dev set evaluation"
    dev_logp, dev_ppl = network.Test(dev_x, dev_y, dev_lang_feat, dev_lang_vec)
    print "Dev cost mean:", dev_logp, "perplexity:", dev_ppl

  if args.test_path:
    print "Test set evaluation"
    test_logp, test_ppl = network.Test(test_x, test_y, test_lang_feat, test_lang_vec)
    print "Test cost mean:", test_logp, "perplexity:", test_ppl


if __name__ == '__main__':
    main()
