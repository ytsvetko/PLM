import os
import cPickle

def Dump(network, message, dump_path):
  # Write components
  for l, layer in enumerate(network):
    for p, param in enumerate(layer.params):
      param_path = os.path.join(dump_path, "%d_%d.pkl" % (l, p))
      cPickle.dump(param.get_value(), open(param_path, 'w'))
  # Write information
  messages_path = os.path.join(dump_path, "messages.txt")
  with open(messages_path, "a") as f:
    f.write(message + "\n")

def Load(network, dump_path):
  # Load components
  for l, layer in enumerate(network):
    for p, param in enumerate(layer.params):
      param_path = os.path.join(dump_path, "%d_%d.pkl" % (l, p))
      param.set_value(cPickle.load(open(param_path, 'r')))

if __name__ == "__main__":
  print "This is a library!"
