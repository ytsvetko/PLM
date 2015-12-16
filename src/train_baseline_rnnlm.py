#!/usr/bin/env python

import sys
import argparse
import codecs
import os
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
print "Loading model"

import baseline_rnnlm as mplm
import symbol_table as st
print "Starting"
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--lang_list', default="sw_ar")
parser.add_argument('--train_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/train/pron-dict.")
parser.add_argument('--dev_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.")
parser.add_argument('--vector_size', type=int, default=100)
parser.add_argument('--context_size', type=int, default=3) # 3 on left
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/baseline_rnnlm")
parser.add_argument('--out_vectors', default="vectors")
parser.add_argument('--load_network_dir')
parser.add_argument('--save_network', action='store_true', default=False)
parser.add_argument('--symbol_table', default="/usr1/home/ytsvetko/projects/mnlm/work/symbol_table.en_ru_fr_ro_it_mt_sw_hi_ar")
parser.add_argument('--lang_vec_file_prefix', default="/usr1/home/ytsvetko/projects/mnlm/data/typology/feat.")
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
      
    w_x.append([symbol_table.WordIndex(word) for word in ngram[:-1]]) # no language vector
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
  try:
    os.stat(os.path.join(args.network_dir, args.lang_list))
  except:
    os.makedirs(os.path.join(args.network_dir, args.lang_list))
    
  symbol_table = st.SymbolTable()
  if os.path.exists(args.symbol_table):
    symbol_table.LoadFromFile(args.symbol_table)

  lang_list = args.lang_list.split("_")
  lang_vec_dict = LoadLangVec(args.lang_vec_file_prefix, lang_list)
  train_x, train_y, train_lang, train_lang_vec = None, None, None, None
  dev_x, dev_y, dev_lang, dev_lang_vec = None, None, None, None
  for lang in lang_list:
    print "Language:", lang
    sys.stdout.flush()
    train_x, train_y, train_lang, train_lang_vec = AppendLangData(
        symbol_table, args.context_size, args.train_path + lang, lang_vec_dict[lang],
        train_x, train_y, lang, train_lang, train_lang_vec)
    dev_x, dev_y, dev_lang, dev_lang_vec = AppendLangData(
        symbol_table, args.context_size, args.dev_path + lang, lang_vec_dict[lang],
        dev_x, dev_y, lang, dev_lang, dev_lang_vec)
  
  all_lang_symbol_indexes = GetLanguageList(args.symbol_table.split(".")[-1].split("_"), symbol_table)
  network = mplm.MNLM(symbol_table.Size(), args.vector_size, args.context_size, all_lang_symbol_indexes)
  if args.load_network_dir:
    network.LoadModel(args.load_network_dir)
  print "Training"
  prev_train_ppl = 100
  prev_dev_ppl = 100
  for epoch in xrange(args.num_epochs):
    print "Epoch:", epoch+1
    train_logp, train_ppl = network.TrainEpoch(train_x, train_y, train_lang, train_lang_vec,
                                               args.batch_size, lr=0.01, to_shuffle=True)
    print "Train cost mean:", train_logp, "perplexity:", train_ppl
    sys.stdout.flush()
    dev_logp, dev_ppl = network.Test(dev_x, dev_y, dev_lang, dev_lang_vec)
    print "Dev cost mean:", dev_logp, "perplexity:", dev_ppl
    sys.stdout.flush()
    if not os.path.exists(os.path.join(args.network_dir, args.lang_list, str(epoch+1))):
      os.mkdir(os.path.join(args.network_dir, args.lang_list, str(epoch+1)))
    
    if args.out_vectors:
      out_vectors_path = os.path.join(args.network_dir, args.lang_list, 
                                      str(epoch+1), args.out_vectors)
      SaveVectors(symbol_table, network.vectors, out_vectors_path)
            
    if args.save_network:
      network.SaveModel(os.path.join(args.network_dir, args.lang_list, str(epoch+1)))

if __name__ == '__main__':
    main()
