import pickle
from conf import beijing_route_hparams
from utils import *
from model import *
import torch
from torch import optim
import numpy as np
import random
import os
import torch.nn.functional as F

def evaluate(eta_model, test_loc_set, test_time_set, hparams, adj_tensor):
  mses = []
  for loc_batch, time_batch in zip(test_loc_set, test_time_set):
    loc_batch = np.array(loc_batch)
    time_batch = np.array(time_batch)  
    loc_batch_tensor = torch.tensor(loc_batch, dtype = torch.long, device = hparams.device)
    pred = eta_model(loc_batch_tensor, adj_tensor)
    label = (time_batch[:, -1, 2] - time_batch[:, 1, 2]) / 3600.0
    label = torch.tensor(label, dtype = torch.float, device = hparams.device)
    mse = torch.mean(label - pred)
    mses.append(mse.item())
  return np.mean(mses)

def test_route_plan(model, test_loc_set, hparams):
  right = 0
  sum_num = 0
  pred_right = 0
  recall_right = 0
  pred_sum = 0
  recall_sum = 0
 
  for batch in test_loc_set:
    batch = np.array(batch)
    input_tra = batch[:, :2]
    label = batch[:, 2:-1]
    des = batch[:, -1]
    pred_batch = []
    des = torch.tensor(des, dtype=torch.long, device = hparams.device)
    for step in range(batch.shape[1] - 3):
      input_tra_tensor = torch.tensor(input_tra, dtype=torch.long, device = hparams.device)  

      pred = model(input_tra_tensor, des)    
      pred = pred.view(pred.shape[1], pred.shape[0], pred.shape[2])
      pred_loc = torch.argmax(pred, 2).tolist()
      pred_loc = np.array(pred_loc)[:, -1]
      pred_batch.append(pred_loc)
      input_tra = np.concatenate((np.array(input_tra.tolist()), pred_loc[:, np.newaxis]), 1)
 
    pred_batch = input_tra[:, 2:]#np.array(pred_batch).transpose()
    for tra_pred, tra_label in zip(pred_batch.tolist(), label.tolist()):
      for item in tra_pred:
        if item in tra_label:
          pred_right += 1
      for item in tra_label:
        if item in tra_pred:
          recall_right += 1          
      pred_sum += len(tra_pred)
      recall_sum += len(tra_label)            
  precision = float(pred_right) / pred_sum
  recall = float(recall_right) / recall_sum
  f1 = (2 * precision * recall) /(precision + recall)

#    for item1, item2 in zip(pred_loc.reshape(-1).tolist(), label.reshape(-1).tolist()):
#      if item1 == item2:
#        right += 1
#      sum_num += 1               
#  print("route plan @acc:", float(right)/sum_num)      
  print("p/r/f:", precision, recall, f1)
 
def train_gru_loc_pred():        
  hparams = dict_to_object(beijing_hparams)
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

  lp_model = LocPredGruModel(hparams, lane_feature, type_feature, length_feature, node_feature).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.gae_epoch):
    print("epoch", i)  
    count = 0
    for batch in train_loc_set:
      model_optimizer.zero_grad()  
      if len(batch[0]) < 4: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device = hparams.device)  
      pred = lp_model(input_tra)
#      pred = pred.permute(1, 0, 2)
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 2000 == 0 and (not count == 0):
        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_loc_pred(lp_model, test_loc_set, hparams, 10)  
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_loc_pred(lp_model, test_loc_set, hparams, 1)  
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        print("step ", str(count))
        print(loss.item())
        torch.save(lp_model.state_dict(), "/data/wuning/NTLR/beijing/model/lp.model_" + str(i))
      count += 1

def train_gcn_loc_pred():        
  hparams = dict_to_object(beijing_hparams)
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
      if len(batch[0]) < 4: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device = hparams.device)  
      pred = lp_model(input_tra)
#      pred = pred.permute(1, 0, 2)
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
#      if count % 2000 == 0 and (not count == 0):
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
#        test_loc_pred(lp_model, test_loc_set, hparams, 10)  
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_loc_pred(lp_model, test_loc_set, hparams, 1)  
        print("step ", str(count))
        print("new gcn : ")
        print(loss.item())
#        torch.save(lp_model.state_dict(), "/data/wuning/NTLR/beijing/model/lp.model_" + str(i))
      count += 1




def train_gat_loc_pred():        
  hparams = dict_to_object(beijing_hparams)
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
      if len(batch[0]) < 4: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device = hparams.device)  
      pred = lp_model(input_tra)
#      pred = pred.permute(1, 0, 2)
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_loc_pred(lp_model, test_loc_set, hparams, 1)  
        print("step ", str(count))
        print("gat:")
        print(loss.item())
#        torch.save(lp_model.state_dict(), "/data/wuning/NTLR/beijing/model/lp.model_" + str(i))
      count += 1



def train_route_plan():
  hparams = dict_to_object(beijing_route_hparams)
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

  lp_model = RoutePlanModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, struct_assign, fnc_assign).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.gae_epoch):
    print("epoch", i)  
    count = 0
    for batch in train_loc_set:
      model_optimizer.zero_grad()  
      if len(batch[0]) < 4: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device = hparams.device)  
      des = torch.tensor(np.array(batch)[:, -1], dtype=torch.long, device = hparams.device)
      pred_label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device = hparams.device)  
      pred = lp_model(input_tra, des)
#      pred = pred.permute(1, 0, 2)
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_route_plan(lp_model, test_loc_set, hparams)  
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        print("step ", str(count))
        print(loss.item())
#        torch.save(lp_model.state_dict(), "/data/wuning/NTLR/beijing/model/lp.model_" + str(i))
      count += 1



def setup_seed(seed):
  torch.manual_seed(seed)  
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
  setup_seed(42)
#  train_struct_cmt()  #get struct assign by autoencoder
#  train_fnc_cmt_loc()  #get fnc assign by graph2seq
#  train_fnc_cmt_rst() #get fnc assign by autoencoder -> reconstruct transition graph  
  train_route_plan()  # three stage model for loc prediction
#  train_gat_loc_pred() # gat baseline model
#  train_gcn_loc_pred()  # gcn baseline model
#  train_gru_loc_pred() # gru baseline model


