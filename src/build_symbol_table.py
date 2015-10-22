#!/usr/bin/env python

import argparse
import codecs
import os
import symbol_table as st

parser = argparse.ArgumentParser()
parser.add_argument('--network_dir', default="/usr1/home/ytsvetko/projects/mnlm/work")
parser.add_argument('--lang_list', default="en_ru_fr_ro_it_mt")
parser.add_argument('--train_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/train/pron-dict.")
parser.add_argument('--dev_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.")
parser.add_argument('--test_path', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.")
parser.add_argument('--lang_vector_path', default="/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.")
parser.add_argument('--symbol_table', default="symbol_table.")
parser.add_argument('--learn_lang', action='store_true', default=False)
parser.add_argument('--brown_clusters')

args = parser.parse_args()

start_symbol = "<s>"
end_symbol = "</s>"

def AddToSymbolTable(corpus, symbol_table):
  symbol_table.WordIndex(start_symbol)
  symbol_table.WordIndex(end_symbol)
  for line in codecs.open(corpus, "r", "utf-8"):
    for word in line.split():
      symbol_table.WordIndex(word)
  
def LoadClusters(filename):
  clusters = {}
  clusters[start_symbol] = "C_EOS"
  clusters[end_symbol] = "C_EOS"

  for line in codecs.open(filename, "r", "utf-8"):
    tokens = line.split()

    if len(tokens) != 3:
      continue
    cluster_id, word, _ = tokens
    clusters[word] = "C_"+cluster_id
  return clusters
 
def main():
  try:
    os.stat(os.path.join(args.network_dir, args.lang_list))
  except:
    os.mkdir(os.path.join(args.network_dir, args.lang_list))
    
  symbol_table = st.SymbolTable()
  symbol_table_path = os.path.join(args.network_dir, args.symbol_table + args.lang_list)
  if os.path.exists(symbol_table_path):
    symbol_table.LoadFromFile(symbol_table_path)
  for lang in args.lang_list.split("_"):
    print "Language:", lang
    if args.learn_lang:
      symbol_table.WordIndex(lang)
    if args.brown_clusters:
      clusters = LoadClusters(args.brown_clusters)
      for phone, cluster_id in clusters.items():
        symbol_table.WordIndex(cluster_id)
    AddToSymbolTable(args.train_path + lang, symbol_table)
    AddToSymbolTable(args.dev_path + lang, symbol_table)
    AddToSymbolTable(args.test_path + lang, symbol_table)
  if args.symbol_table:
    symbol_table.SaveToFile(symbol_table_path)

if __name__ == '__main__':
    main()
