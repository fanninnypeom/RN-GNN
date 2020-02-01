import pickle
from conf import beijing_label_hparams
from utils import *
from model import *
import torch
from torch import optim
import numpy as np
import random
import os
import torch.nn.functional as F

def test_label_pred(model, test_set, test_label, hparams):
  right = 0
  sum_num = 0
  test_set = torch.tensor(test_set, dtype=torch.long, device = hparams.device)  
  pred = model(test_set)    
  pred_loc = torch.argmax(pred, 1).tolist()
  right_pos = 0
  right_neg = 0
  wrong_pos = 0
  wrong_neg = 0
  for item1, item2 in zip(pred_loc, test_label):
    if item1 == item2:
      right += 1
      if item2 == 1:
        right_pos += 1
      else:  
        right_neg += 1      
    else:
      if item2 == 1:
        wrong_pos += 1
      else:  
        wrong_neg += 1          
    sum_num += 1
  recall = float(right_pos)/(right_pos + wrong_pos)                
  precision = float(right_pos)/(wrong_neg + right_pos)
  f1 = 2*recall*precision/(precision + recall)
  print("label prediction @acc @p/r/f:", float(right)/sum_num, precision, recall, f1)      

def train_gat_label_pred():
  hparams = dict_to_object(beijing_label_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, struct_assign, fnc_assign = load_label_pred_data(hparams) 
  ce_criterion = torch.nn.CrossEntropyLoss()

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

  lp_model = LabelPredGatModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, struct_assign, fnc_assign).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.label_epoch):
    print("epoch", i)  
    count = 0
    train_set, train_label, test_set, test_label = get_label_train_data(hparams)
    model_optimizer.zero_grad()  
    train_set = torch.tensor(train_set, dtype=torch.long, device = hparams.device)  
    train_label = torch.tensor(train_label, dtype=torch.long, device = hparams.device)  
    pred = lp_model(train_set)
    print(pred.shape, train_label.shape, train_set.shape)
    loss = ce_criterion(pred, train_label)
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
    model_optimizer.step()
    if count % 20 == 0:
      test_label_pred(lp_model, test_set, test_label, hparams)  
      print("step ", str(count))
      print(loss.item())
#      torch.save(lp_model.state_dict(), "/data/wuning/RN-GNN/beijing/model/label_pred.model_" + str(i))
    count += 1



def train_gcn_label_pred():
  hparams = dict_to_object(beijing_label_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, struct_assign, fnc_assign = load_label_pred_data(hparams) 
  ce_criterion = torch.nn.CrossEntropyLoss()

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

  lp_model = LabelPredGcnModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, struct_assign, fnc_assign).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.label_epoch):
    print("epoch", i)  
    count = 0
    train_set, train_label, test_set, test_label = get_label_train_data(hparams)
    model_optimizer.zero_grad()  
    train_set = torch.tensor(train_set, dtype=torch.long, device = hparams.device)  
    train_label = torch.tensor(train_label, dtype=torch.long, device = hparams.device)  
    pred = lp_model(train_set)
    print(pred.shape, train_label.shape, train_set.shape)
    loss = ce_criterion(pred, train_label)
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
    model_optimizer.step()
    if count % 20 == 0:
      test_label_pred(lp_model, test_set, test_label, hparams)  
      print("step ", str(count))
      print(loss.item())
#      torch.save(lp_model.state_dict(), "/data/wuning/RN-GNN/beijing/model/label_pred.model_" + str(i))
    count += 1



def train_label_pred():
  hparams = dict_to_object(beijing_label_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, struct_assign, fnc_assign = load_label_pred_data(hparams) 
  ce_criterion = torch.nn.CrossEntropyLoss()

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

  lp_model = LabelPredModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, struct_assign, fnc_assign).to(hparams.device)

  model_optimizer = optim.Adam(lp_model.parameters(), lr=hparams.lp_learning_rate)

  for i in range(hparams.label_epoch):
    print("epoch", i)  
    count = 0
    train_set, train_label, test_set, test_label = get_label_train_data(hparams)
    model_optimizer.zero_grad()  
    train_set = torch.tensor(train_set, dtype=torch.long, device = hparams.device)  
    train_label = torch.tensor(train_label, dtype=torch.long, device = hparams.device)  
    pred = lp_model(train_set)
    print(pred.shape, train_label.shape, train_set.shape)
    loss = ce_criterion(pred, train_label)
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(lp_model.parameters(), hparams.lp_clip)
    model_optimizer.step()
    if count % 20 == 0:
      test_label_pred(lp_model, test_set, test_label, hparams)  
      print("step ", str(count))
      print(loss.item())
#      torch.save(lp_model.state_dict(), "/data/wuning/RN-GNN/beijing/model/label_pred.model_" + str(i))
    count += 1

def setup_seed(seed):
  torch.manual_seed(seed)  
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
  setup_seed(42)
  train_label_pred()  # three stage model for loc prediction
#  train_gat_label_pred() # gat baseline model
#  train_gcn_label_pred()  # gcn baseline model
#  train_gru_loc_pred() # gru baseline model


