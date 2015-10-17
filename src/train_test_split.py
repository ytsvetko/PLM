#!/usr/bin/env python3

import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--in_corpus_file', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.sw")
parser.add_argument('--out_train_dir', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/train")
parser.add_argument('--out_dev_dir', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/dev")
parser.add_argument('--out_test_dir', default="/usr1/home/ytsvetko/projects/mnlm/data/pron/test")
parser.add_argument('--dev_ratio', type=float, default=0.15)
parser.add_argument('--test_ratio', type=float, default=0.10)
parser.add_argument('--random_seed', type=int, default=2015)
args = parser.parse_args()

def main():
  random.seed(args.random_seed)
  filename = os.path.basename(args.in_corpus_file)
  test_file = open(os.path.join(args.out_test_dir, filename), "w")
  train_file = open(os.path.join(args.out_train_dir, filename), "w")
  dev_file = open(os.path.join(args.out_dev_dir, filename), "w")
  for line in open(args.in_corpus_file):
    tokens = line.split(" ||| ")
    if len(tokens) != 2: 
      continue
    pronunciation = tokens[1]
    if random.random() < args.dev_ratio:
      dev_file.write(pronunciation)
    elif random.random() < args.test_ratio:
      test_file.write(pronunciation)
    else:
      train_file.write(pronunciation)

if __name__ == '__main__':
    main()
