#!/usr/bin/env python3

import argparse
import sys
import numpy
import math
import collections
import json
from scipy.stats.stats import pearsonr
from scipy import spatial

import levenshtein


parser = argparse.ArgumentParser()
parser.add_argument('--src_tgt_pairs')
parser.add_argument('--src_vectors')
parser.add_argument('--tgt_vectors')
parser.add_argument('--num_closest', type=int)
parser.add_argument('--out_filename', default=sys.stdout)
args = parser.parse_args()

def Similarity(v1, v2, metric="cosine"):
  def IsZero(v):
    return all(n == 0 for n in v)    

  if metric == "correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return pearsonr(v1, v2)[0]

  if metric == "abs_correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return abs(pearsonr(v1, v2)[0])

  if metric == "cosine":
    return spatial.distance.cosine(v1, v2)
    
def LoadVectors(filename):    
  pronunciations = collections.defaultdict(list)
  vectors_r = collections.defaultdict(list)
  vectors_softmax = collections.defaultdict(list)
  for line in open(filename):
    tokens = line.strip().split(" ||| ")
    if len(tokens) != 4:
      continue
    word, pron, word_vector_softmax, word_vector_r = tokens
    pronunciations[word].append(pron)      
    vectors_r[pron].append(word_vector_r)      
    vectors_softmax[pron].append(word_vector_softmax)      
  return pronunciations, vectors_r, vectors_softmax

def Flatten(l):
  return [item for sublist in l for item in sublist]
      
def GetVector(char_list, str_vectors):      
  def StrToArray(str_vector):
    return [float(num) for num in  str_vector.split()]
    
  def VectorList(str_vectors):
    result = []
    for str_vector in str_vectors.split("\t"):
      result.append(StrToArray(str_vector))
    return result
          
  vector = []
  vector_list = VectorList(str_vectors)
  next_vector = 0
  char_vector_size = len(str_vectors.split("\t")[0].split())
  for char in char_list:
    if char == "":
      vector.append([1]*char_vector_size)
    else:
      vector.append(vector_list[next_vector])
      next_vector += 1
  return numpy.array(Flatten(vector))
  
def FindClosest(src_pronunciations, tgt_pronunciations, src_vectors, tgt_vectors, num_closest):
  closest = []
  for src_pron in src_pronunciations:
    for tgt_word, tgt_pron_list in tgt_pronunciations.items():
      for tgt_pron in tgt_pron_list:
        src_aligned, tgt_aligned, err_list, optimal_cost = levenshtein.Distance().AlignStrings(src_pron.split(), tgt_pron.split())
        src_vector = GetVector(src_aligned, src_vectors[src_pron][0])
        tgt_vector = GetVector(tgt_aligned, tgt_vectors[tgt_pron][0])
        assert src_vector.shape == tgt_vector.shape, (src_vector.shape, tgt_vector.shape) 
        closest.append((Similarity(src_vector, tgt_vector), tgt_word))  
  result = []
  result_set = set()
  for (distance, word) in sorted(closest, reverse=False):
    if len(result) ==  num_closest:
      break
    if word in result_set:
      continue
    result_set.add(word)
    result.append(word)
  return result
  
def main():
  src_pronunciations, src_vectors_r, src_vectors_softmax = LoadVectors(args.src_vectors)
  tgt_pronunciations, tgt_vectors_r, tgt_vectors_softmax = LoadVectors(args.tgt_vectors)

  if args.out_filename == sys.stdout:
    out_f = sys.stdout
  else:
    out_f = open(args.out_filename, "w")

  correct = 0
  total = 0
    
  for line in open(args.src_tgt_pairs):
    en, src_words, tgt_words = line.strip().split(" ||| ")
    src_words = src_words.split()
    tgt_words = tgt_words.split()
    src_word_set = set(src_words)
    tgt_word_set = set(tgt_words)
    for w in src_words:
      if w not in src_pronunciations:
        src_word_set.remove(w)
    for w in tgt_words: 
      if w not in tgt_pronunciations:
        tgt_word_set.remove(w) 
    if len(src_word_set) == 0 or len(tgt_word_set) == 0:
      continue
    total += 1
    src_word_set_pronunciations = []
    for w in src_word_set:
      src_word_set_pronunciations.append(src_pronunciations[w])
    src_word_set_pronunciations = Flatten(src_word_set_pronunciations) 
    tgt_closest_words_r = FindClosest(src_word_set_pronunciations, tgt_pronunciations, 
                                  src_vectors_r, tgt_vectors_r, args.num_closest)
    #tgt_closest_words_r = FindClosest(src_word_set_pronunciations, tgt_pronunciations, 
    #                              src_vectors_softmax, tgt_vectors_softmax, args.num_closest)
    found = False
    for tgt_word_i in tgt_word_set:
      if tgt_closest_words_r[0] == tgt_word_i:
        found = True
        correct += 1
    out_f.write("{} ||| {} ||| {}\n".format(" ".join(src_words), " ".join(tgt_closest_words_r), found))
    sys.stdout.flush()
  out_f.write("Total: {}, Correct: {}, Accuracy: {}\n".format(total, correct, correct/total))
    
if __name__=='__main__':
  main()

