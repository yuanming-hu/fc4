from utils import print_angular_errors
import cPickle as pickle
import sys
import os
from utils import *

def load_errors(model_name):
  model_path = 'models/fc4/' + model_name + '/'
  pkl = list(sorted(filter(lambda x: x.startswith('error'), os.listdir(model_path))))[-1]
  with open(os.path.join(model_path, pkl)) as f:
    return pickle.load(f)

def combine(models):
  combined = []
  for model in models:
    combined += load_errors(model)
  return combined

if __name__ == '__main__':
  models = sys.argv[1:]
  print_angular_errors(combine(models))
