import json
import math
from math import sin, cos, sqrt, atan2, radians
import pickle
import numpy as np
import pandas as pd
from numpy import random

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

locList = pickle.load(open("/data/wuning/NTLR/beijing/locList","rb"))
roadNet = json.load(open("/data/wuning/map-matching/osmextract-node.json","r"))

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

tsp_set = []
tsp_label = []

node_features = []
for i in range(len(locList)):
  node = id2road[int(locList[i])]
  coords = node["geometry"]["coordinates"]
  length = calc_length(coords)
  lanes = 1
  if "lanes" in node["properties"]:
    lanes = int(node["properties"]["lanes"])
  node_features.append([lanes, length, i])# node["properties"][highway]
 # print("type:", node["properties"]["highway"])
#  print(node["properties"])
#  if "bridge" in node["properties"]:
#    print("bridge")  
  if "bridge" in node["properties"]:
#    print(node["properties"]["bridge"])
    tsp_set.append(i)   
    tsp_label.append(1) 
#    print(i)
tsp_false_set = []
while(len(tsp_false_set) < 100):
  node = random.randint(16000 - 1)    
  if (not node in tsp_set) and (not node in tsp_false_set):
    tsp_false_set.append(node)   
      

label_pred_train_set = []

  
node_features = np.array(node_features)
type_set = pd.Series(node_features[:, 1])
labels, levels = pd.factorize(type_set)
node_features[:, 1] = labels
node_features[:, 2] = (node_features[:, 2].astype(np.float) / 0.01).astype(np.int)

node_features = node_features.astype(np.int)

pickle.dump(node_features, open("/data/wuning/RN-GNN/beijing/node_features_nh", "wb"))

pickle.dump(tsp_set, open("/data/wuning/RN-GNN/beijing/label_pred_train_set", "wb"))

pickle.dump(tsp_false_set, open("/data/wuning/RN-GNN/beijing/label_pred_train_set_false", "wb"))

#pickle.dump(tsp_set, open("/data/wuning/RN-GNN/beijing/label_pred_train_set", "wb"))
#length dividede by 0.01  230 id 
