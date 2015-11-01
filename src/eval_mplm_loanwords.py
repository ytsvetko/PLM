#!/usr/bin/env python3

import argparse
import os, sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--loanwords_test', default="/usr1/home/ytsvetko/projects/mnlm/data/loanwords/sw-ar/test.en-sw-ar")
parser.add_argument('--loanwords_log', default="/usr1/home/ytsvetko/projects/mnlm/work/mplm_loanwords/sw_ar_nbest.out")
args = parser.parse_args()

def EvalTestSet(loanwords_filename, log_filename):
  eval_dict = {}
  for line in open(loanwords_filename):
    tokens = line.strip().split(" ||| ")
    en, src_words, gold_tgt_words = tokens[0], tokens[1], tokens[2]
    src_words = src_words.split()
    gold_tgt_words = set(gold_tgt_words.split())
    if len(src_words) == 0 or len(gold_tgt_words) == 0:
      continue
    for src_word in src_words:
      eval_dict[src_word] = gold_tgt_words

  total = 0
  correct = 0
  for line in open(log_filename):
    [src_word, best_tgt_words] = json.loads(line)
    if src_word in eval_dict:
      total +=1
      if best_tgt_words[0][1] in eval_dict[src_word]:
        correct += 1
  print("Total: {}, Accuracy: {}".format(total, correct/total)) 

def main():   
  EvalTestSet(args.loanwords_test, args.loanwords_log) 

if __name__ == '__main__':
    main()
