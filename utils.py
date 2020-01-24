import pickle
import numpy as np
from scipy import sparse
import random
import copy

EPS = 1e-30

def gen_false_edge(adj, num):
#  adj = copy.deepcopy(adj)
  adj = adj.todense()
  edges = []
  while len(edges) < num:
    start = random.randint(0, len(adj) - 1)      
    end = random.randint(0, len(adj) - 1)   
    if adj[start, end] == 0 and adj[end, start] == 0:
      edges.append([start, end])    
  edges = np.array(edges)
  edges = edges.transpose()
  return edges

def load_g2s_data(hparams):
  adj = pickle.load(open(hparams.adj, "rb"))
  self_loop = np.eye(len(adj))
  adj = np.array(adj) + self_loop
  adj = sparse.coo_matrix(adj)
  node_features = pickle.load(open(hparams.node_features, "rb"))
  node_features = node_features.tolist()
  while len(node_features) < 16000:
    node_features.append(['0', '0', '0', '0'])
  node_features = np.array(node_features)

  spectral_label = pickle.load(open(hparams.spectral_label, "rb"))
  train_cmt_set = pickle.load(open(hparams.train_cmt_set, "rb"))

  return adj, node_features, spectral_label, train_cmt_set


def load_gae_data(hparams):
  adj = pickle.load(open(hparams.adj, "rb"))
  self_loop = np.eye(len(adj))
  adj = np.array(adj) + self_loop
  adj = sparse.coo_matrix(adj)
  node_features = pickle.load(open(hparams.node_features, "rb"))
  node_features = node_features.tolist()
  while len(node_features) < 16000:
    node_features.append(['0', '0', '0', '0'])
  node_features = np.array(node_features)

  spectral_label = pickle.load(open(hparams.spectral_label, "rb"))

  return adj, node_features, spectral_label

def load_data(hparams):

  train_loc_set = pickle.load(open(hparams.train_loc_set, "rb"))
  train_time_set = pickle.load(open(hparams.train_time_set, "rb"))
  adj = pickle.load(open(hparams.adj, "rb"))
  self_loop = np.eye(len(adj))
  adj = np.array(adj) + self_loop
  adj = sparse.coo_matrix(adj)
  test_loc_set = train_loc_set[6000:]
  test_time_set = train_time_set[6000:]
  train_loc_set = train_loc_set[:6000]
  train_time_set = train_time_set[:6000]
  return train_loc_set, train_time_set, adj, test_loc_set, test_time_set

class Dict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def dict_to_object(dictObj):
  if not isinstance(dictObj, dict):
    return dictObj
  inst=Dict()
  for k,v in dictObj.items():
    inst[k] = dict_to_object(v)
  return inst


