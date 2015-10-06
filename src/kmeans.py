#!/usr/bin/env python3

import argparse
import sys
import numpy
import math
import collections

from scipy.cluster.vq import kmeans2, kmeans, whiten

parser = argparse.ArgumentParser()
parser.add_argument('--vectors')
parser.add_argument('--num_clusters', type=int)
parser.add_argument('--out_clusters', default=sys.stdout)
args = parser.parse_args()

def LoadVectors(filename):    
  words = []
  vectors = []
  vector_size = 0
  for line in open(filename):
    tokens = line.split()
    word = tokens[0]
    vector = numpy.array([float(num) for num in  tokens[1:]])
    if vector_size == 0:
      vector_size = len(vector)
    else:
      assert vector_size == len(vector), (vector_size, len(vector))
    """ normalize weight vector """
    vector /= math.sqrt((vector**2).sum())
    vectors.append(vector)
    words.append(word)      
  return words, vectors

def KNN(vectors, n):
  print (len(vectors), n)
  vectors = whiten(vectors)
  centroid, label = kmeans2(vectors, n, minit='points')
  return centroid, label
  
def main():
  words, vectors = LoadVectors(args.vectors)
  centroid, label = KNN(vectors, args.num_clusters)
  clusters = collections.defaultdict(list)
  for word_ind, cluster in enumerate(label):
    clusters[cluster].append(words[word_ind])

  if args.out_clusters == sys.stdout:
    out_f = sys.stdout
  else:
    out_f = open(args.out_clusters, "w")
  for cluster_ind, cluster in clusters.items():
    out_f.write("{} ||| ".format(cluster_ind))
    for word in cluster:
      out_f.write("{} ".format(word))
    out_f.write("\n")
    
if __name__=='__main__':
  main()

