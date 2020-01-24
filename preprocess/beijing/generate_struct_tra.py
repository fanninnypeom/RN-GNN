import pickle
import numpy as np

fi = open("/data/wuning/NTLR/beijing/train_loc_set", "rb")
fs = open("sparse_spectral_labels", "rb")
train_cmt_set = []
train_loc_set = pickle.load(fi)
sp_label = pickle.load(fs)
for batch in train_loc_set:
  cmt_batch = []
  for tra in batch:
    cmt_tra = []
    last = None
    for item in tra:
      if not sp_label[item] == last:  
        cmt_tra.append(sp_label[item])        
        last = sp_label[item]
    cmt_batch.append(cmt_tra)  
  train_cmt_set.append(cmt_batch)
print(np.array(train_cmt_set).shape)  
pickle.dump(train_cmt_set, open("/data/wuning/RN-GNN/beijing/train_cmt_tra", "wb"))



