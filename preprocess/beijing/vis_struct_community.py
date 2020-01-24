import numpy as np
import pickle
import json
locList = pickle.load(open("/data/wuning/NTLR/beijing/locList","rb"))
roadNet = json.load(open("/data/wuning/map-matching/osmextract-node.json","r"))
fo = open("/data/wuning/RN-GNN/beijing/main_road.js", "w")
roads = []
id2cor = {}
for item in roadNet["features"]:
  id2cor[item["properties"]["id"]] = item["geometry"]["coordinates"] 

t = pickle.load(open("/home/wuning/RN-GNN/struct_assign", "rb"))
t = np.array(t)
t = t.transpose()
print(t.shape)
t = t.argsort()[:,::-1][:,:30]
print(t.shape)
coords = []

for i in range(10):
  cmts = []  
  for r_id in t[i]:    
    if r_id < len(locList):
      cmts.append(id2cor[int(locList[r_id])])      
  coords.append(cmts)

fo = open("/data/wuning/RN-GNN/beijing/main_road.js", "w")
fo.write(str(coords))
