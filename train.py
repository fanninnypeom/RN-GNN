import pickle
from conf import beijing_hparams
from utils import *
from model import *
import torch
from torch import optim
import numpy as np
import random
import os

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
  adj_tensor = torch.tensor(adj, dtype = torch.long, device = hparams.device)
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
  train()  
    

