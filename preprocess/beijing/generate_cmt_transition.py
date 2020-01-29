import pickle
f = open("/data/wuning/RN-GNN/beijing/train_cmt_set", "rb")
train_cmt_set = pickle.load(f)
skips = 5
cmt_adj = [[0 for j in range(300)] for i in range(300)]
for batch in train_cmt_set:
  for tra in batch:
    for i in range(len(tra)):
      for j in range(1, skips + 1):  
        if i + j < len(tra):  
          cmt_adj[tra[i]][tra[i + j]] += 1       

pickle.dump(cmt_adj, open("/data/wuning/RN-GNN/beijing/cmt_tra_adj", "wb"))                    

