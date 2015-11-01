#!/usr/bin/env python

import argparse
import codecs
import os, sys
import numpy
import collections
from sklearn.cross_validation import train_test_split

import mplm_context as mplm
import symbol_table as st

import levenshtein

parser = argparse.ArgumentParser()
parser.add_argument('--loanwords_training', default="/usr1/home/ytsvetko/projects/mnlm/data/loanwords/mt-it/train.en-mt-it")
parser.add_argument('--src_pronunciations', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.mt")
parser.add_argument('--tgt_pronunciations', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.it")
parser.add_argument('--src_lang', default="mt")
parser.add_argument('--tgt_lang', default="it")

parser.add_argument('--context_size', type=int, default=2)
parser.add_argument('--vector_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/mplm_loanwords")
parser.add_argument('--out_vectors', default="vectors")
parser.add_argument('--load_network', action='store_true', default=False)
parser.add_argument('--load_network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords/mt_it/100")
parser.add_argument('--save_network', action='store_true', default=False)
parser.add_argument('--symbol_table', default="/usr1/home/ytsvetko/projects/mnlm/work/symbol_table.en_ru_fr_ro_it_mt_sw_hi_ar")
args = parser.parse_args()

start_symbol = u"<s>"
end_symbol = u"</s>"

insert_symbol = u"INS"
delete_symbol = u"DEL"

def LoadPronunciations(filename):
  pronunciations = collections.defaultdict(list)
  for line in codecs.open(filename, "r", "utf-8"):
    tokens = line.strip().split(" ||| ")
    if len(tokens) != 2:
      continue
    word, pronunciation = tokens
    pronunciations[word].append(pronunciation)
  return pronunciations

def LoadData(loanwords_filename, src_pronunciations, tgt_pronunciations, 
             context_size, symbol_table, src_lang, tgt_lang):
  x = []
  y = []
  lang_symbols = []

  for line in codecs.open(loanwords_filename, "r", "utf-8"):
    tokens = line.strip().split(" ||| ")
    en, src_words, tgt_words = tokens[0], tokens[1], tokens[2]
    src_words = src_words.split()
    tgt_words = tgt_words.split()
    if len(src_words) == 0 or len(tgt_words) == 0:
      continue
    for src_word in src_words:
      for tgt_word in tgt_words:
        if src_word not in src_pronunciations or tgt_word not in tgt_pronunciations:
          print "Missing pronunciation", line
          sys.stdout.flush()
          continue
        for src_pronunciation in src_pronunciations[src_word]:
          src_pronunciation = src_pronunciation.split()
          for tgt_pronunciation in tgt_pronunciations[tgt_word]:
            tgt_pronunciation = tgt_pronunciation.split()
            src_aligned, tgt_aligned, err_list, optimal_cost = levenshtein.Distance().AlignStrings(src_pronunciation, tgt_pronunciation)
            assert len(src_aligned) == len(tgt_aligned), (src_aligned, tgt_aligned)
            assert len(src_aligned) == len(err_list), (src_aligned, err_list)
            src_aligned = [start_symbol]*context_size + src_aligned + [end_symbol]*context_size
            src_aligned = [insert_symbol if w == "" else w for w in src_aligned ]
            tgt_aligned = [delete_symbol if w == "" else w for w in tgt_aligned ]
            sys.stdout.flush()
            for i, _ in enumerate(tgt_aligned):
              x_vector = [symbol_table.WordIndex(word) for word in src_aligned[i:i+context_size*2+1]]
              x_vector.append(symbol_table.WordIndex(src_lang))
              assert len(x_vector) == (context_size*2+2), (src_aligned, tgt_aligned, i, x_vector)
              x.append(x_vector)
              y.append(symbol_table.WordIndex(tgt_aligned[i]))
              lang_symbols.append(symbol_table.WordIndex(tgt_lang))
  return numpy.array(x), numpy.array(y), numpy.array(lang_symbols)
   
def GetLanguageList(languages, symbol_table):
  return numpy.array([symbol_table.WordIndex(l) for l in languages])

def SaveVectors(symbol_table, vector_matrix, filename):
  out_f = codecs.open(filename, "w", "utf-8")
  for i, vector in enumerate(vector_matrix):
    vector = [str(num) for num in vector]
    out_f.write(u"{} {}\n".format(symbol_table.IndexToWord(i), " ".join(vector)))

def main():
  lang_pair = args.src_lang+"_"+args.tgt_lang
  try:
    os.stat(os.path.join(args.network_dir, lang_pair))
  except:
    os.makedirs(os.path.join(args.network_dir, lang_pair))
    
    
  symbol_table = st.SymbolTable()
  if os.path.exists(args.symbol_table):
    symbol_table.LoadFromFile(args.symbol_table)

  src_pronunciations = LoadPronunciations(args.src_pronunciations)
  tgt_pronunciations = LoadPronunciations(args.tgt_pronunciations)

  x, y, lang_symbols = LoadData(args.loanwords_training,
                              src_pronunciations, tgt_pronunciations, 
                              args.context_size, symbol_table, 
                              args.src_lang, args.tgt_lang)

  train_x, dev_x, train_y, dev_y, train_lang, dev_lang,  = train_test_split(x, y, lang_symbols, test_size=0.15, random_state=123)
  print "Training size", len(train_x), len(train_y), len(train_lang)
  print "Dev size", len(dev_x), len(dev_y), len(dev_lang)
  sys.stdout.flush()
  all_lang_symbol_indexes = GetLanguageList(args.symbol_table.split(".")[-1].split("_"), symbol_table)
  print symbol_table.Size(), args.vector_size, 2+args.context_size*2, all_lang_symbol_indexes
  network = mplm.MNLM(symbol_table.Size(), args.vector_size, 2+args.context_size*2, all_lang_symbol_indexes)
  print "Network loaded"
  if args.load_network:
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
    if not os.path.exists(os.path.join(args.network_dir, lang_pair, str(epoch+1))):
      os.mkdir(os.path.join(args.network_dir, lang_pair, str(epoch+1)))
    
    if args.out_vectors:
      out_vectors_path = os.path.join(args.network_dir, lang_pair, 
                                      str(epoch+1), args.out_vectors)
      SaveVectors(symbol_table, network.vectors, out_vectors_path)
            
    if args.save_network:
      network.SaveModel(os.path.join(args.network_dir, lang_pair, str(epoch+1)))

if __name__ == '__main__':
    main()
