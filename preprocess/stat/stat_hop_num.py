from networkx import nx
import random
import pickle
import numpy as np
graphData = pickle.load(open("/data/wuning/map-matching/allGraph", "rb"))
stats = []
adj = np.matrix(graphData)[:15500, :15500]
G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())

for i in range(1000):
  start = random.randint(0, 15000)
  end = random.randint(0, 15000)
  try:
    shortest_path_length = nx.shortest_path_length(G, source=start, target=end) 
  except:
    continue
  print(shortest_path_length)
