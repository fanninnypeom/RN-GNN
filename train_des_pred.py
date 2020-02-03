import pickle
from conf import beijing_hparams
from utils import *
from model import *
import torch
from torch import optim
import numpy as np
import random
import os
import torch.nn.functional as F


def test_des_pred(model, test_loc_set, hparams):
  right = 0
  sum_num = 0
  for batch in test_loc_set:
    batch = np.array(batch)
    if len(batch[0]) < 10:
      continue    
    input_tra = batch[:, :-10]
    label = batch[:, -1]

    input_tra = torch.tensor(input_tra, dtype=torch.long, device = hparams.device)  
    label_tensor = torch.tensor(label, dtype=torch.long, device = hparams.device)  

    pred = model(input_tra)    
    pred = pred.view(pred.shape[1], pred.shape[0], pred.shape[2])
    pred_loc = torch.argmax(pred, 2).tolist()
    pred_loc = np.array(pred_loc)[:, -1]

    for item1, item2 in zip(pred_loc.tolist(), label.tolist()):
      if item1 == item2:
        right += 1
      sum_num += 1
                
  print("des prediction @acc:", float(right)/sum_num)      

def train_gat_des_pred():
  hparams = dict_to_object(beijing_des_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, struct_assign, fnc_assign, train_loc_set = load_loc_pred_data(hparams) 
  ce_criterion = torch.nn.CrossEntropyLoss()
  train_loc_set = train_loc_set[ : -500]
  test_loc_set = train_loc_set[-500 : ]
  adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1), dtype=torch.long).t()
  adj_values = torch.tensor(adj.data, dtype=torch.float)
  adj_shape = adj.shape
  adj_tensor = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)

  features = features.astype(np.int)

  lane_feature = torch.tensor(features[:, 0], dtype=torch.long, device = hparams.device)
  type_feature = torch.tensor(features[:, 1], dtype=torch.long, device = hparams.device)
  length_feature = torch.tensor(features[:, 2], dtype=torch.long, device = hparams.device)
  node_feature = torch.tensor(features[:, 3], dtype=torch.long, device = hparams.device)
 
  struct_assign = torch.tensor(struct_assign, dtype=torch.float, device = hparams.device)
  fnc_assign = torch.tensor(fnc_assign, dtype=torch.float, device = hparams.device)

  lp_model = LocPredGatModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.gae_epoch):
    print("epoch", i)  
    count = 0
    for batch in train_loc_set:
      model_optimizer.zero_grad()  
      if len(batch[0]) < 12: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-10], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, -1], dtype=torch.long, device = hparams.device)  
      pred = lp_model(input_tra)
      pred = pred.view(pred.shape[1], pred.shape[0], pred.shape[2])
      pred = pred[:, -1, :]
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_des_pred(lp_model, test_loc_set, hparams)  
        print("step ", str(count))
        print(loss.item())
      count += 1



def train_gcn_des_pred():
  hparams = dict_to_object(beijing_des_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, struct_assign, fnc_assign, train_loc_set = load_loc_pred_data(hparams) 
  ce_criterion = torch.nn.CrossEntropyLoss()
  train_loc_set = train_loc_set[ : -500]
  test_loc_set = train_loc_set[-500 : ]
  adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1), dtype=torch.long).t()
  adj_values = torch.tensor(adj.data, dtype=torch.float)
  adj_shape = adj.shape
  adj_tensor = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)

  features = features.astype(np.int)

  lane_feature = torch.tensor(features[:, 0], dtype=torch.long, device = hparams.device)
  type_feature = torch.tensor(features[:, 1], dtype=torch.long, device = hparams.device)
  length_feature = torch.tensor(features[:, 2], dtype=torch.long, device = hparams.device)
  node_feature = torch.tensor(features[:, 3], dtype=torch.long, device = hparams.device)
 
  struct_assign = torch.tensor(struct_assign, dtype=torch.float, device = hparams.device)
  fnc_assign = torch.tensor(fnc_assign, dtype=torch.float, device = hparams.device)

  lp_model = LocPredGcnModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.gae_epoch):
    print("epoch", i)  
    count = 0
    for batch in train_loc_set:
      model_optimizer.zero_grad()  
      if len(batch[0]) < 12: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-10], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, -1], dtype=torch.long, device = hparams.device)  
      pred = lp_model(input_tra)
      pred = pred.view(pred.shape[1], pred.shape[0], pred.shape[2])
      pred = pred[:, -1, :]
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_des_pred(lp_model, test_loc_set, hparams)  
        print("step ", str(count))
        print(loss.item())
      count += 1


def train_des_pred():
  hparams = dict_to_object(beijing_des_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, struct_assign, fnc_assign, train_loc_set = load_loc_pred_data(hparams) 
  ce_criterion = torch.nn.CrossEntropyLoss()
  train_loc_set = train_loc_set[ : -500]
  test_loc_set = train_loc_set[-500 : ]
  adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1), dtype=torch.long).t()
  adj_values = torch.tensor(adj.data, dtype=torch.float)
  adj_shape = adj.shape
  adj_tensor = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)

  features = features.astype(np.int)

  lane_feature = torch.tensor(features[:, 0], dtype=torch.long, device = hparams.device)
  type_feature = torch.tensor(features[:, 1], dtype=torch.long, device = hparams.device)
  length_feature = torch.tensor(features[:, 2], dtype=torch.long, device = hparams.device)
  node_feature = torch.tensor(features[:, 3], dtype=torch.long, device = hparams.device)
 
  struct_assign = torch.tensor(struct_assign, dtype=torch.float, device = hparams.device)
  fnc_assign = torch.tensor(fnc_assign, dtype=torch.float, device = hparams.device)

  lp_model = LocPredModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, struct_assign, fnc_assign).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.gae_epoch):
    print("epoch", i)  
    count = 0
    for batch in train_loc_set:
      model_optimizer.zero_grad()  
      if len(batch[0]) < 12: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-10], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, -1], dtype=torch.long, device = hparams.device)  
      pred = lp_model(input_tra)
      pred = pred.view(pred.shape[1], pred.shape[0], pred.shape[2])
      pred = pred[:, -1, :]
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_des_pred(lp_model, test_loc_set, hparams)  
        print("step ", str(count))
        print(loss.item())
      count += 1

def setup_seed(seed):
  torch.manual_seed(seed)  
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
  setup_seed(42)
#  train_des_pred()  # three stage model for des prediction
  train_gat_des_pred() # gat baseline model
#  train_gcn_des_pred()  # gcn baseline model
#  train_gru_loc_pred() # gru baseline model


