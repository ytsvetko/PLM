#!/usr/bin/env python

import json
import codecs

class SymbolTable(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = []

  def LoadFromFile(self, filename):
    self.index_to_word = json.load(codecs.open(filename, 'r', 'utf-8'))
    for i, word in enumerate(self.index_to_word):
      self.word_to_index[word] = i
      
  def SaveToFile(self, filename):
    json.dump(self.index_to_word, codecs.open(filename, 'w', 'utf-8'))

  def WordIndex(self, word):
    if word not in self.word_to_index:
      self.word_to_index[word] = len(self.index_to_word)
      self.index_to_word.append(word)
    return self.word_to_index[word]

  def IndexToWord(self, ind):
    return self.index_to_word[ind]

  def Size(self):
    return len(self.index_to_word)
  
if __name__ == "__main__":
  print "This is a library!"
