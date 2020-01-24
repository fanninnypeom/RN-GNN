import json
import pickle
import numpy as np
import networkx as nx
import itertools
from sklearn.cluster import SpectralClustering
from scipy import sparse

adj = pickle.load(open("/data/wuning/NTLR/beijing/CompleteAllGraph", "rb"))
adj = np.array(adj)

adj = sparse.coo_matrix(adj)

sc = SpectralClustering(300, affinity='precomputed', n_init=1, assign_labels="discretize")    

sc.fit(adj)

pickle.dump(sc.labels_, open("sparse_spectral_labels", "wb"))
print(sc.labels_)

for i in range(300):
  print(np.where(sc.labels_ == i)[0].shape)    
