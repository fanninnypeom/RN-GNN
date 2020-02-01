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

def test_loc_pred(model, test_loc_set, hparams, steps):
  right = 0
  sum_num = 0
  for batch in test_loc_set:
    batch = np.array(batch)
    input_tra = batch[:, :-steps]
    label = batch[:, -1]
    for step in range(steps):
      input_tra = torch.tensor(input_tra, dtype=torch.long, device = hparams.device)  
      label_tensor = torch.tensor(label, dtype=torch.long, device = hparams.device)  

      pred = model(input_tra)    
      pred = pred.view(pred.shape[1], pred.shape[0], pred.shape[2])
      pred_loc = torch.argmax(pred, 2).tolist()
      pred_loc = np.array(pred_loc)[:, -1]

      input_tra = np.concatenate((np.array(input_tra.tolist()), pred_loc[:, np.newaxis]), 1)
    for item1, item2 in zip(pred_loc.tolist(), label.tolist()):
      if item1 == item2:
        right += 1
      sum_num += 1
                
  print(str(steps) + "step prediction @acc:", float(right)/sum_num)      

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
      if count % 2000 == 0 and (not count == 0):
        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_loc_pred(lp_model, test_loc_set, hparams, 10)  
      if count % 200 == 0:
#        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        test_loc_pred(lp_model, test_loc_set, hparams, 1)  
        print("step ", str(count))
        print("new gcn : ")
        print(loss.item())
        torch.save(lp_model.state_dict(), "/data/wuning/NTLR/beijing/model/lp.model_" + str(i))
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
        torch.save(lp_model.state_dict(), "/data/wuning/NTLR/beijing/model/lp.model_" + str(i))
      count += 1



def train_loc_pred():
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

  lp_model = LocPredModel(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, struct_assign, fnc_assign).to(hparams.device)

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
        test_loc_pred(lp_model, test_loc_set, hparams, 5)  
        print("step ", str(count))
        print(loss.item())
        torch.save(lp_model.state_dict(), "/data/wuning/NTLR/beijing/model/lp.model_" + str(i))
      count += 1

def train_fnc_cmt_rst():   #train fnc by reconstruction
  hparams = dict_to_object(beijing_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, struct_assign, t_adj = load_loc_rst_data(hparams) 
  ce_criterion = torch.nn.MSELoss()
 
  adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1), dtype=torch.long).t()
  adj_values = torch.tensor(adj.data, dtype=torch.float)
  adj_shape = adj.shape
  adj_tensor = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)

  t_adj_indices = torch.tensor(np.concatenate([t_adj.row[:, np.newaxis], t_adj.col[:, np.newaxis]], 1), dtype=torch.long).t()
  t_adj_values = torch.tensor(t_adj.data, dtype=torch.float)
  t_adj_shape = t_adj.shape
  t_adj_tensor = torch.sparse.FloatTensor(t_adj_indices, t_adj_values, t_adj_shape)

  features = features.astype(np.int)

  lane_feature = torch.tensor(features[:, 0], dtype=torch.long, device = hparams.device)
  type_feature = torch.tensor(features[:, 1], dtype=torch.long, device = hparams.device)
  length_feature = torch.tensor(features[:, 2], dtype=torch.long, device = hparams.device)
  node_feature = torch.tensor(features[:, 3], dtype=torch.long, device = hparams.device)
  struct_assign = torch.tensor(struct_assign, dtype=torch.float, device = hparams.device)

  g2t_model = GraphAutoencoderTra(hparams).to(hparams.device)

  model_optimizer = optim.Adam(g2t_model.parameters(), lr=hparams.g2t_learning_rate)

  for i in range(hparams.g2t_epoch):
    print("epoch", i)
    input_edge, label = generate_edge(t_adj, hparams.g2t_sample_num)  
    label = torch.tensor(label, dtype=torch.float, device = hparams.device)
    input_edge = torch.tensor(input_edge, dtype=torch.long, device = hparams.device)
    pred = g2t_model(lane_feature, type_feature, length_feature, node_feature, adj_tensor, t_adj_tensor, struct_assign, input_edge)
    count = 0

    loss = ce_criterion(pred, label)
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(g2t_model.parameters(), hparams.g2t_clip)
    model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
    if count % 10 == 0:
      print(loss.item())
      torch.save(g2t_model.state_dict(), "/data/wuning/RN-GNN/beijing/model/g2t.model_" + str(i))
      pickle.dump(g2t_model.fnc_assign.tolist(), open("/data/wuning/RN-GNN/beijing/fnc_assign", "wb"))
      count += 1

def train_fnc_cmt_loc():
  hparams = dict_to_object(beijing_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, main_assign, train_cmt_set = load_g2s_loc_data(hparams) 
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
  main_assign = torch.tensor(main_assign, dtype=torch.float, device = hparams.device)

  g2s_model = Graph2SeqLoc(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, main_assign).to(hparams.device)

  model_optimizer = optim.Adam(g2s_model.parameters(), lr=hparams.g2s_learning_rate)

  for i in range(hparams.gae_epoch):
    print("epoch", i)  
    count = 0
    for batch in train_cmt_set:
      model_optimizer.zero_grad()  
      if len(batch[0]) < 4: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device = hparams.device)  
      pred = g2s_model(input_tra)
#      pred = pred.permute(1, 0, 2)
      loss = ce_criterion(pred.view(-1, hparams.node_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(g2s_model.parameters(), hparams.g2s_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 200 == 0:
        print(loss.item())
        torch.save(g2s_model.state_dict(), "/data/wuning/NTLR/beijing/model/g2s.model_" + str(i))
        pickle.dump(g2s_model.fnc_assign.tolist(), open("/data/wuning/RN-GNN/beijing/fnc_assign", "wb"))
         
      count += 1



def train_fnc_cmt():
  hparams = dict_to_object(beijing_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, main_assign, train_cmt_set = load_g2s_data(hparams) 
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
  main_assign = torch.tensor(main_assign, dtype=torch.float, device = hparams.device)

  g2s_model = Graph2SeqCmt(hparams, lane_feature, type_feature, length_feature, node_feature, adj_tensor, main_assign).to(hparams.device)

  model_optimizer = optim.Adam(g2s_model.parameters(), lr=hparams.g2s_learning_rate)

  for i in range(hparams.gae_epoch):
    print("epoch", i)  
    count = 0
    for batch in train_cmt_set:
      model_optimizer.zero_grad()  
      if len(batch[0]) < 4: 
        continue     
      input_tra = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device = hparams.device)  
      pred_label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device = hparams.device)  
      pred = g2s_model(input_tra)
#      pred = pred.permute(1, 0, 2)
      loss = ce_criterion(pred.view(-1, hparams.struct_cmt_num), pred_label.view(-1))
      loss.backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(g2s_model.parameters(), hparams.g2s_clip)
      model_optimizer.step()
#      print("grad:", g2s_model.linear.weight.grad)
      if count % 200 == 0:
        print(loss.item())
        torch.save(g2s_model.state_dict(), "/data/wuning/NTLR/beijing/model/g2s.model_" + str(i))
      count += 1


def train_struct_cmt():
  hparams = dict_to_object(beijing_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  adj, features, sp_label = load_gae_data(hparams) 
  gae_model = GraphAutoencoder(hparams).to(hparams.device)
  mse_criterion = torch.nn.MSELoss()
  ce_criterion = torch.nn.BCELoss()
  model_optimizer = optim.Adam(gae_model.parameters(), lr=hparams.gae_learning_rate)
  adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1), dtype=torch.long).t()
  adj_values = torch.tensor(adj.data, dtype=torch.float)
  adj_shape = adj.shape
  adj_tensor = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)
  features = features.astype(np.int)
  lane_feature = torch.tensor(features[:, 0], dtype=torch.long, device = hparams.device)
  type_feature = torch.tensor(features[:, 1], dtype=torch.long, device = hparams.device)
  length_feature = torch.tensor(features[:, 2], dtype=torch.long, device = hparams.device)
  node_feature = torch.tensor(features[:, 3], dtype=torch.long, device = hparams.device)
  assign_label = torch.tensor(sp_label, dtype=torch.float, device = hparams.device)
  
  for i in range(hparams.gae_epoch):
    model_optimizer.zero_grad()  
    f_edge = gen_false_edge(adj, adj.row.shape[0])  
    f_edge = torch.tensor(f_edge, dtype=torch.long, device = hparams.device)
    print("epoch", i)  
    edge_h, struct_adj, pred_cmt_adj, main_assign, edge_e, edge_label = gae_model(adj_tensor, lane_feature, type_feature, length_feature, node_feature, f_edge)
#    loss = mse_criterion(struct_adj, pred_cmt_adj)
    print("edge_e:", torch.mean(edge_e[:10000]), torch.mean(edge_e[-10000:]))
    ce_loss = ce_criterion(edge_e, edge_label)
    
    loss = ce_criterion(F.softmax(main_assign, 1), assign_label)

    loss.backward(retain_graph=True)
    ce_loss.backward()
#    print("dec grad:", torch.sum(gae_model.dec_gnn.cmt_gat_0.weight.grad, 1), gae_model.dec_gnn.cmt_gat_0.weight.grad.shape)
#    print("grad:", gae_model.enc_gnn.cmt_gat_0.a.grad)

    torch.nn.utils.clip_grad_norm_(gae_model.parameters(), hparams.clip)
    model_optimizer.step()
    print(ce_loss.item())
    if i % 50 == 0:
      pickle.dump(main_assign.tolist(), open("struct_assign", "wb"))  
      torch.save(gae_model.state_dict(), "/data/wuning/NTLR/beijing/model/gae.model_" + str(i))

def train():
  hparams = dict_to_object(beijing_hparams)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
  hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_loc_set, train_time_set, adj, test_loc_set, test_time_set = load_data(hparams) 
  eta_model = ETAModelMLP(hparams).to(hparams.device)
  torch.save(eta_model.state_dict(), "/data/wuning/RN-GNN/beijing/model/rn_gcn_eta.model")
#  eta_model = torch.nn.DataParallel(eta_model, device_ids=[0,1,2])
#  eta_model = eta_model.cuda()
  criterion = torch.nn.MSELoss()
  model_optimizer = optim.Adam(eta_model.parameters(), lr=hparams.eta_learning_rate)
  count = 0
  adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1), dtype=torch.long).t()
  adj_values = torch.tensor(adj.data, dtype=torch.float)
  adj_shape = adj.shape
  adj_tensor = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)
#  adj_tensor = torch.tensor(adj, dtype = torch.long, device = hparams.device)
  train_loc_set.extend(test_loc_set)
  train_time_set.extend(test_time_set)
  for i in range(hparams.eta_epoch):
    erres = []  
    for loc_batch, time_batch in zip(train_loc_set, train_time_set):
      model_optimizer.zero_grad()  
      loc_batch = np.array(loc_batch)
      time_batch = np.array(time_batch)  
      if not len(loc_batch) == 100:
        continue  

      label = (time_batch[:, -1, 2] - time_batch[:, 1, 2]) / 3600.0
      if label.max() > 1.0:
        continue

      loc_batch_tensor = torch.tensor(loc_batch, dtype = torch.long, device = hparams.device)
      pred = eta_model(loc_batch_tensor, adj_tensor)
      label = torch.tensor(label, dtype = torch.float, device = hparams.device)
      loss = criterion(pred, label)
      print(loss.item())
      if count > 500:
        erres.append(loss.item())
      else:
        loss.backward()
        model_optimizer.step()

#      print(eta_model.embedding.grad)
#      print("emb grad:", np.mean(eta_model.embedding.grad.cpu().numpy(), 1))
#      print("influence node:", 16000 - sum(np.mean(eta_model.embedding.grad.cpu().numpy(), 1) == 0))
      count += 1
#      if count % 100 == 0:
#        print("count: ", count)    
    torch.save(eta_model.state_dict(), "/data/wuning/NTLR/beijing/model/rn_gcn_eta.model_" + str(i))
    print("test_loss:", np.mean(np.array(erres)))

#      mse = evaluate(eta_model, test_loc_set, test_time_set, hparams, adj_tensor)
#      print("test_loss:", mse)

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
#  train_loc_pred()  # three stage model for loc prediction
#  train_gat_loc_pred() # gat baseline model
  train_gcn_loc_pred()  # gcn baseline model
#  train_gru_loc_pred() # gru baseline model


