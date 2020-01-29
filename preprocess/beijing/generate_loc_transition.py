import pickle
f = open("/data/wuning/NTLR/beijing/train_loc_set", "rb")
train_loc_set = pickle.load(f)
skips = 5
loc_adj = [[0 for j in range(16000)] for i in range(16000)]
for batch in train_loc_set:
  for tra in batch:
    for i in range(len(tra)):
      for j in range(1, skips + 1):  
        if i + j < len(tra):  
          loc_adj[tra[i]][tra[i + j]] += 1       
          if not tra[i] == tra[i + j]:
            loc_adj[tra[i + j]][tra[i]] += 1       


pickle.dump(loc_adj, open("/data/wuning/RN-GNN/beijing/loc_tra_adj", "wb"))                    

