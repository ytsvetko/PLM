#!/usr/bin/env python3

"""
./precompute_levenshtein.py --loanwords_training /usr1/home/ytsvetko/projects/mnlm/data/loanwords/ro-fr/train.en-ro-fr --src_pronunciations /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.ro --tgt_pronunciations /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.fr --out_path /usr1/home/ytsvetko/projects/mnlm/work/levenshtein/ro-fr

./precompute_levenshtein.py --loanwords_training /usr1/home/ytsvetko/projects/mnlm/data/loanwords/mt-it/train.en-mt-it --src_pronunciations /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.mt --tgt_pronunciations /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.it --out_path /usr1/home/ytsvetko/projects/mnlm/work/levenshtein/mt-it

./precompute_levenshtein.py --loanwords_training /usr1/home/ytsvetko/projects/mnlm/data/loanwords/mt-it/test.en-mt-it --src_pronunciations /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.mt --tgt_pronunciations /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.it --out_path /usr1/home/ytsvetko/projects/mnlm/work/levenshtein/mt-it

"""
import argparse
import levenshtein
import os
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--loanwords_training', default="/usr1/home/ytsvetko/projects/mnlm/data/loanwords/sw-ar/train.en-sw-ar")
parser.add_argument('--src_pronunciations', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.sw")
parser.add_argument('--tgt_pronunciations', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.ar")
parser.add_argument('--out_path', default="/usr1/home/ytsvetko/projects/mnlm/work/levenshtein/sw-ar")

args = parser.parse_args()

def LoadPronunciations(filename):
  pronunciations = collections.defaultdict(list)
  for line in open(filename):
    tokens = line.strip().split(" ||| ")
    if len(tokens) != 2:
      continue
    word, pronunciation = tokens
    pronunciations[word].append(pronunciation)
  return pronunciations

def Levenshtein(loanwords_filename, src_pronunciations, tgt_pronunciations):
  x = []
  y = []
  def contains_sublist(lst, sublst):
    n = len(sublst)
    return any((sublst == lst[i:i+n]) for i in xrange(len(lst)-n+1))

  for line in open(loanwords_filename):
    tokens = line.strip().split(" ||| ")
    en, src_words, tgt_words = tokens[0], tokens[1], tokens[2]
    src_words = src_words.split()
    tgt_words = tgt_words.split()
    if len(src_words) == 0 or len(tgt_words) == 0:
      continue
    for src_word in src_words:
      if src_word not in src_pronunciations:
        print("Missing pronunciation", line)
        continue
      for src_pronunciation in src_pronunciations[src_word]:
        src_pronunciation = src_pronunciation.split()
        for tgt_word in tgt_pronunciations:
          for tgt_pronunciation in tgt_pronunciations[tgt_word]:
            tgt_pronunciation = tgt_pronunciation.split()
            src_aligned, tgt_aligned, err_list, optimal_cost = levenshtein.Distance().AlignStrings(src_pronunciation, tgt_pronunciation)
            yield (src_word, " ".join(src_pronunciation), 
                   tgt_word , " ".join(tgt_pronunciation), 
                   src_aligned, tgt_aligned, err_list, optimal_cost)             

def main():
  src_pronunciations = LoadPronunciations(args.src_pronunciations)
  tgt_pronunciations = LoadPronunciations(args.tgt_pronunciations)

  for (src_word, src_pronunciation, 
       tgt_word, tgt_pronunciation, 
       src_aligned, tgt_aligned, 
       err_list, optimal_cost) in Levenshtein(args.loanwords_training,
                                              src_pronunciations, tgt_pronunciations):
    out_f = open(os.path.join(args.out_path, src_word), "a")
    out_f.write("{} ||| {} ||| {} ||| {} ||| {} ||| {} ||| {}\n".format(src_pronunciation, 
       tgt_word, tgt_pronunciation, src_aligned, tgt_aligned, err_list, optimal_cost))
                              
if __name__ == '__main__':
    main()
