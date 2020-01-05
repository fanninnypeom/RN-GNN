import pickle

EPS = 1e-30

def load_data(hparams):

  train_loc_set = pickle.load(open(hparams.train_loc_set, "rb"))
  train_time_set = pickle.load(open(hparams.train_time_set, "rb"))
  adj = pickle.load(open(hparams.adj, "rb"))

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


