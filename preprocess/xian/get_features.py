import json
import math
from math import sin, cos, sqrt, atan2, radians
import pickle
import numpy as np
import pandas as pd

def get_distance(lat1, lon1, lat2, lon2):
  lat1 = radians(lat1)
  lon1 = radians(lon1)
  lat2 = radians(lat2)
  lon2 = radians(lon2)
  R = 6373.0
  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))

  distance = R * c

  return distance

locList = pickle.load(open("/data/wuning/RN-GNN/xian/locList","rb"))
roadNet = json.load(open("/data/wuning/RN-GNN/xian/network.json","r"))

features = ["lane", "type", "length", "id"]

id2road = {}
for item in roadNet["features"]:
  id2road[int(item["properties"]["id"])] = item


def calc_length(coords):
  length = 0
  last_cd = coords[0]
  for i in range(1, len(coords)):
    cd = coords[i]  
    length += get_distance(cd[1], cd[0], last_cd[1], last_cd[0])  
    last_cd = cd
  return length

node_features = []
for i in range(len(locList)):
  node = id2road[int(locList[i])]
  coords = node["geometry"]["coordinates"]
  length = calc_length(coords)
  lanes = 1
  if "lanes" in node["properties"]:
    lanes = int(node["properties"]["lanes"])
  node_features.append([lanes, node["properties"]["highway"], length, i])

node_features = np.array(node_features)
type_set = pd.Series(node_features[:, 1])
labels, levels = pd.factorize(type_set)
node_features[:, 1] = labels
node_features[:, 2] = (node_features[:, 2].astype(np.float) / 0.01).astype(np.int)

node_features = node_features.astype(np.int)

pickle.dump(node_features, open("/data/wuning/RN-GNN/xian/node_features", "wb"))

#length dividede by 0.01  230 id 
