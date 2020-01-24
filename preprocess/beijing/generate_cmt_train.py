import pickle
import time
import numpy as np
history = {}
cmt_batches = []

fd = open("/data/wuning/RN-GNN/beijing/train_cmt_tra", "rb")
cmtData = pickle.load(fd)
print("cmtData:", np.array(cmtData).shape)
cmts = []

for bat_1 in cmtData:
  cmts.extend(bat_1)

print("cmts:", len(cmts))

lenIndexedCmt = {}

for cmt in zip(cmts):
  cmt = cmt[0]  
  length = len(cmt)
  if length in lenIndexedCmt:
    lenIndexedCmt[length].append(cmt[:length])   
  else:
    lenIndexedCmt[length] = [cmt[:length]]
                  

cmt_batches = []

for key in lenIndexedCmt:
  print(key, len(lenIndexedCmt[key]))  
  for i in range(0, len(lenIndexedCmt[key]), 100):    
    cmt_batches.append(lenIndexedCmt[key][i : i + 100])          


fd = open("/data/wuning/RN-GNN/beijing/train_cmt_set", "wb")

pickle.dump(cmt_batches, fd, -1)

