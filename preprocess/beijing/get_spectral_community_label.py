import pickle
sps_cmt_label = pickle.load(open("sparse_spectral_labels", "rb"))
one_hot_label = []
for item in sps_cmt_label:
  a = [0 for i in range(300)]
  a[item] = 1
  one_hot_label.append(a)

pickle.dump(one_hot_label, open("/data/wuning/RN-GNN/beijing/spectral_label", "wb"))      


