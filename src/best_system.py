#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', default="en")
args = parser.parse_args()

def EpochResults(filename):
  for line in open(filename):
    if line.startswith("Epoch:"):
      epoch = line.split()[-1]
    if line.startswith("Dev cost mean:"):
      ppl = line.split()[-1]
      #print(("Epoch", epoch, "Perplexity", str(ppl)))
      yield (epoch, float(ppl))
    
def main():
  min_ppl = 100
  epoch = None
  for e, p in EpochResults(args.log_file):
    if p < min_ppl: 
      epoch, min_ppl = e, p
  print("Epoch", epoch, "Perplexity", str(min_ppl))
  
if __name__ == '__main__':
    main()
