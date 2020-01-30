import pickle
import numpy as np
import json
locList = pickle.load(open("/data/wuning/RN-GNN/chengdu/locList","rb"))
roadNet = json.load(open("/data/wuning/RN-GNN/chengdu/extract-chengdu-feat.json","r"))

sourceNode = []
endNode = []
adjItems = []
one_way = []
roadAdj = [[0 for i in range(3000)] for j in range(3000)]
ID2road = {}
for item in roadNet["features"]:
  if not str(item["properties"]["class"]) == "LNLink":
    continue    
  ID2road[item["properties"]["id"]] = item
for item in locList:
  try:
    sourceNode.append(ID2road[item]["geometry"]["from"]) 
    endNode.append(ID2road[item]["geometry"]["to"])
  except:
    sourceNode.append("abc")
    endNode.append("def")
    continue
  if("oneway" in ID2road[item]["properties"] and (ID2road[item]["properties"]["oneway"] == "yes")):
    one_way.append(len(endNode) - 1)  
print("source:", sourceNode)
print("end:", endNode)
sourceNode = np.array(sourceNode)
endNode = np.array(endNode) 
   
for i in range(len(locList)):
#  if i % 100 == 0:
#    print(i)    
  index = np.where(sourceNode == endNode[i])
  if not index in one_way:  
    m_index = np.where(endNode == sourceNode[i]) 
    index = index[0].tolist()
    index.extend(m_index[0].tolist())  
  for item in index:
      roadAdj[i][item] = 1  

rawGraph = open("/data/wuning/RN-GNN/chengdu/CompleteAllGraph", "wb")
pickle.dump(roadAdj, rawGraph, -1)












  
