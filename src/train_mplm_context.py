#!/usr/bin/env python

import sys
import argparse
import codecs
import os
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
print "Loading model"
import mplm_context as mplm
import symbol_table as st
print "Starting"
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--lang_list', default="sw_ar")
parser.add_argument('--train_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/train/pron-dict.")
parser.add_argument('--dev_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.")
parser.add_argument('--vector_size', type=int, default=100)
parser.add_argument('--context_size', type=int, default=2) # 2 on left, 2 on right
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang")
parser.add_argument('--out_vectors', default="vectors")
parser.add_argument('--load_network_dir')
parser.add_argument('--save_network', action='store_true', default=False)
parser.add_argument('--symbol_table', default="/usr1/home/ytsvetko/projects/mnlm/work/symbol_table.en_ru_fr_ro_it_mt_sw_hi_ar")
args = parser.parse_args()

start_symbol = "<s>"
end_symbol = "</s>"

insert_symbol = "INS"
delete_symbol = "DEL"


def LoadData(corpus, symbol_table, context_size, lang):
  # in format: text corpus, embeddings, n-gram order
  # out format: for each n-gram, x: n-1 embeddings appended; y: n's word 1-hot representation
  lang_symbol = symbol_table.WordIndex(lang)
  x = []
  y = []
  lang_symbols = []
  def next_ngram(corpus):
    for line in codecs.open(corpus, "r", "utf-8"):
      tokens = line.split()
      #pad with boundary tokens
      tokens = [start_symbol]*context_size + tokens + [end_symbol]*context_size
      for ngram in zip(*[tokens[i:] for i in range(1+context_size*2)]):
        yield ngram
       
  for ngram in next_ngram(corpus):
    x.append([symbol_table.WordIndex(word) for word in ngram[:context_size]] +
             [symbol_table.WordIndex(insert_symbol)] +
             [symbol_table.WordIndex(word) for word in ngram[context_size+1:]] + 
             [lang_symbol])
    y.append(symbol_table.WordIndex(ngram[context_size]))
    lang_symbols.append(lang_symbol)

  return x, y, lang_symbols

def SaveVectors(symbol_table, vector_matrix, filename):
  out_f = codecs.open(filename, "w", "utf-8")
  for i, vector in enumerate(vector_matrix):
    vector = [str(num) for num in vector]
    out_f.write(u"{} {}\n".format(symbol_table.IndexToWord(i), " ".join(vector)))

def AppendLangData(symbol_table, context_size, corpus_filename, 
                   x, y, lang, lang_symbols):
  x_lang, y_lang, lang_symbols_lang = LoadData(corpus_filename, symbol_table, context_size, lang)
  if x is None:
    return numpy.array(x_lang), numpy.array(y_lang), numpy.array(lang_symbols_lang)
  else:
    x = numpy.concatenate((x, x_lang), axis=0)
    y = numpy.concatenate((y, y_lang), axis=0)
    lang_symbols = numpy.concatenate((lang_symbols, lang_symbols_lang), axis=0)
    return x, y, lang_symbols
  
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
  train_x, train_y, train_lang = None, None, None
  dev_x, dev_y, dev_lang = None, None, None
  for lang in args.lang_list.split("_"):
    print "Language:", lang
    sys.stdout.flush()
    train_x, train_y, train_lang = AppendLangData(
        symbol_table, args.context_size, args.train_path + lang,
        train_x, train_y, lang, train_lang)
    dev_x, dev_y, dev_lang = AppendLangData(
        symbol_table, args.context_size, args.dev_path + lang,
        dev_x, dev_y, lang, dev_lang)

  all_lang_symbol_indexes = GetLanguageList(args.symbol_table.split(".")[-1].split("_"), symbol_table)
  network = mplm.MNLM(symbol_table.Size(), args.vector_size, 2+args.context_size*2, all_lang_symbol_indexes)
  if args.load_network_dir:
    network.LoadModel(args.load_network_dir)
  print "Training"
  prev_train_ppl = 100
  prev_dev_ppl = 100
  for epoch in xrange(args.num_epochs):
    print "Epoch:", epoch+1
    train_logp, train_ppl = network.TrainEpoch(train_x, train_y, train_lang, args.batch_size, lr=0.01)
    print "Train cost mean:", train_logp, "perplexity:", train_ppl
    sys.stdout.flush()
    dev_logp, dev_ppl = network.Test(dev_x, dev_y, dev_lang)
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
