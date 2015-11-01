#!/usr/bin/env python

import argparse
import codecs
import os, sys
import numpy
import collections
import json

import mplm_context as mplm
import symbol_table as st

parser = argparse.ArgumentParser()
parser.add_argument('--loanwords_test', default="/usr1/home/ytsvetko/projects/mnlm/data/loanwords/mt-it/test.en-mt-it")
parser.add_argument('--levenshtein_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/levenshtein")
parser.add_argument('--src_lang', default="mt")
parser.add_argument('--tgt_lang', default="it")

parser.add_argument('--context_size', type=int, default=2)
parser.add_argument('--vector_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--load_network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords/mt_it/100")
parser.add_argument('--symbol_table', default="/usr1/home/ytsvetko/projects/mnlm/work/symbol_table.en_ru_fr_ro_it_mt_sw_hi_ar")
parser.add_argument('--output_file')
args = parser.parse_args()

start_symbol = u"<s>"
end_symbol = u"</s>"

insert_symbol = u"INS"
delete_symbol = u"DEL"

def ReadLevenshtein(levenshtein_file, context_size):
  def ParseStrList(str_list):
    str_list = str_list[2:-2]
    return str_list.split("', '")

  for line in codecs.open(levenshtein_file, "r", "utf-8"):
    src_pron, tgt_word, tgt_pron, src_aligned, tgt_aligned, _, _ = line.strip().split(" ||| ")
    src_pron = src_pron.split()
    tgt_pron = tgt_pron.split()
    src_aligned = ParseStrList(src_aligned)
    tgt_aligned = ParseStrList(tgt_aligned)
    src_aligned = [insert_symbol if w == "" else w for w in src_aligned ]
    tgt_aligned = [delete_symbol if w == "" else w for w in tgt_aligned ]
    src_aligned = [start_symbol]*context_size + src_aligned + [end_symbol]*context_size
    yield src_pron, tgt_word, tgt_pron, src_aligned, tgt_aligned

def BuildTestSet(loanwords_filename, levenshtein_dir, context_size, symbol_table, src_lang, tgt_lang):
  levenshtein_dir = os.path.join(levenshtein_dir, src_lang + "-" + tgt_lang)
  x = []
  y = []
  lang_symbols = []
  seen_tuples = set()
  for line in codecs.open(loanwords_filename, "r", "utf-8"):
    tokens = line.strip().split(" ||| ")
    en, src_words, gold_tgt_words = tokens[0], tokens[1], tokens[2]
    src_words = src_words.split()
    gold_tgt_words = gold_tgt_words.split()
    if len(src_words) == 0 or len(gold_tgt_words) == 0:
      continue
    for src_word in src_words:
      print "Loading word:", src_word
      levenshtein_file = os.path.join(levenshtein_dir, src_word)
      if not os.path.exists(levenshtein_file):
        print "No Levenshtein file."
        continue
      for src_pron, tgt_word, tgt_pron, src_aligned, tgt_aligned in ReadLevenshtein(levenshtein_file,
                                                                                    context_size):
        for i, tgt_c in enumerate(tgt_aligned):
          x_vector = [symbol_table.WordIndex(word) for word in src_aligned[i:i+context_size*2+1]]
          x_vector.append(symbol_table.WordIndex(src_lang))
          assert len(x_vector) == (context_size*2+2), (src_aligned, tgt_aligned, i, x_vector)
          key_tuple = (tuple(x_vector), tgt_c)
          if key_tuple in seen_tuples:
            continue
          seen_tuples.add(key_tuple)
          x.append(x_vector)
          y.append(symbol_table.WordIndex(tgt_c))
          lang_symbols.append(symbol_table.WordIndex(tgt_lang))
  return numpy.array(x), numpy.array(y), numpy.array(lang_symbols)

def ProcessResults(loanwords_filename, levenshtein_dir, context_size, symbol_table, 
                   src_lang, tgt_lang, log_probs, out_file, nbest_tgts=10):
  levenshtein_dir = os.path.join(levenshtein_dir, src_lang + "-" + tgt_lang)
  seen_tuples = {}
  for line in codecs.open(loanwords_filename, "r", "utf-8"):
    tokens = line.strip().split(" ||| ")
    en, src_words, gold_tgt_words = tokens[0], tokens[1], tokens[2]
    src_words = src_words.split()
    if len(src_words) == 0 or len(gold_tgt_words) == 0:
      continue
    for src_word in src_words:
      best_tgt_words = []
      print "Loading word:", src_word
      levenshtein_file = os.path.join(levenshtein_dir, src_word)
      if not os.path.exists(levenshtein_file):
        print "No Levenshtein file."
        continue
      for src_pron, tgt_word, tgt_pron, src_aligned, tgt_aligned in ReadLevenshtein(levenshtein_file,
                                                                                    context_size):
        pron_log_prob = 0.0
        for i, tgt_c in enumerate(tgt_aligned):
          x_vector = [symbol_table.WordIndex(word) for word in src_aligned[i:i+context_size*2+1]]
          x_vector.append(symbol_table.WordIndex(src_lang))
          assert len(x_vector) == (context_size*2+2), (src_aligned, tgt_aligned, i, x_vector)
          key_tuple = (tuple(x_vector), tgt_c)
          if key_tuple in seen_tuples:
            log_prob_index = seen_tuples[key_tuple]
          else:
            log_prob_index = len(seen_tuples)
            seen_tuples[key_tuple] = log_prob_index
          pron_log_prob += log_probs[log_prob_index]

        if len(best_tgt_words) == nbest_tgts and best_tgt_words[-1][0] > pron_log_prob:
          continue
        best_tgt_words.append((pron_log_prob, tgt_word))
        best_tgt_words = sorted(best_tgt_words, reverse=True)[:nbest_tgts]
      out_file.write(json.dumps([src_word, best_tgt_words]))
      out_file.write("\n")

def GetLanguageList(languages, symbol_table):
  return numpy.array([symbol_table.WordIndex(l) for l in languages])

def main():   
  symbol_table = st.SymbolTable()
  symbol_table.LoadFromFile(args.symbol_table)

  x, y, lang_symbols = BuildTestSet(args.loanwords_test, args.levenshtein_dir, args.context_size,
                                    symbol_table, args.src_lang, args.tgt_lang)

  print "Test sizes:", x.shape, y.shape, lang_symbols.shape

  sys.stdout.flush()
  all_lang_symbol_indexes = GetLanguageList(args.symbol_table.split(".")[-1].split("_"), symbol_table)
  print symbol_table.Size(), args.vector_size, 2+args.context_size*2, all_lang_symbol_indexes
  
  print "Loading network"
  network = mplm.MNLM(symbol_table.Size(), args.vector_size, 2+args.context_size*2, all_lang_symbol_indexes)
  network.LoadModel(args.load_network_dir)
  print "Network loaded"

  print "Testing"
  log_probs = network.LogProb(x, y, lang_symbols, batch_size=args.batch_size)
  out_file = sys.stdout
  if (args.output_file):
    out_file = codecs.open(args.output_file, "w", "utf-8")
  ProcessResults(args.loanwords_test, args.levenshtein_dir, args.context_size,
                 symbol_table, args.src_lang, args.tgt_lang, log_probs, out_file)
  sys.stdout.flush()

if __name__ == '__main__':
    main()
