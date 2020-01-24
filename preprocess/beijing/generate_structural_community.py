import json
import pickle
import numpy as np
import networkx as nx
import itertools

main_road = ["motorway_link", "trunk_link", "primary_link", "motorway", "trunk", "primary", "secondary", "secondary_link"]#, "secondary"]
roadNet = json.load(open("/data/wuning/map-matching/osmextract-node.json","r"))
locList = pickle.load(open("/data/wuning/NTLR/beijing/locList","rb"))
typeList = []
road2type = {}  # 1 main  0 brh
for road in roadNet["features"]:
  road_id = road["properties"]["id"]  
  if (str(road_id) in locList or road_id in locList) and "highway" in road["properties"] and road["properties"]["highway"] in main_road:
    road2type[road["properties"]["id"]] = 1
  elif (str(road_id) in locList or road_id in locList) and "highway" in road["properties"]:
    road2type[road["properties"]["id"]] = 0 
print(len(road2type.keys()))
print(len(locList))
for loc in locList:
  typeList.append(road2type[int(loc)])
print(typeList)  
pickle.dump(typeList, open("/data/wuning/RN-GNN/beijing/typeList","wb"))     

adj = pickle.load(open("/data/wuning/NTLR/beijing/CompleteAllGraph", "rb"))
adj = np.array(adj)

for i in range(len(typeList)):
  if typeList[i] == 1:
    adj[i, :] = 0
    adj[:, i] = 0  
#    adj[i] = [0 for item in range(len(adj[0]))]
    

for i in range(len(adj)):
  for j in range(len(adj[0])):
    adj[i][j] = adj[j][i]

G = nx.from_numpy_matrix(adj, create_using=nx.Graph())

all_connected_subgraphs = list(nx.connected_component_subgraphs(G))


#for nb_nodes in range(2, G.number_of_nodes()):
#  for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
#    if nx.is_connected(SG):
#      print(SG.nodes)
#      all_connected_subgraphs.append(SG)

pickle.dump(all_connected_subgraphs, open("/data/wuning/RN-GNN/beijing/all_cnt_graph","wb"))


