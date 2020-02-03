import pickle
import numpy as np
f = open("/data/wuning/NTLR/beijing/train_loc_set", "rb")
train_loc_set = pickle.load(f)
train_loc_int_set = []
interval = 5
for batch in train_loc_set:
  int_batch = [] 
  batch = np.array(batch)
  for i in range(0, len(batch[0]), interval):
    int_batch.append(batch[:, i]) 
  train_loc_int_set.append(np.array(int_batch).transpose())
  print("shape_in:", batch.shape, "shape_out", np.array(int_batch).transpose().shape)

pickle.dump(train_loc_int_set, open("/data/wuning/NTLR/beijing/train_loc_" + str(interval) + "_set", "wb"))                    

