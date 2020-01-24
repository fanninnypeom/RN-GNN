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

t = pickle.load(open("sparse_spectral_labels", "rb"))
coords = []
for i in range(300):
  nodes = np.where(t == i)  
  print(nodes)
  cmts = []  
  for r_id in nodes[0].tolist():    
    if r_id < len(locList):
      cmts.append(id2cor[int(locList[r_id])][0])      
  coords.append(cmts)

fo = open("/data/wuning/RN-GNN/beijing/main_road.js", "w")
fo.write(str(coords))
