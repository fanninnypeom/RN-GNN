# coding=utf-8
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from gcn_layers import *
import torch.nn as nn
from cmt_gen import get_cmt
import torch.nn.functional as F


class ETAModelMLP(Module):
  def __init__(self, hparams):
    super(ETAModelMLP, self).__init__()
    self.hparams = hparams
    self.BiGRU = GRU(self.hparams).to(self.hparams.device)

    self.linear = torch.nn.Linear(hparams.node_dims * 2, 1)

    self.embedding_size = hparams.node_dims
    self.vocab_size = hparams.vocab_size
    self.embedding = Parameter(torch.FloatTensor(hparams.node_num, hparams.node_dims))
    torch.nn.init.xavier_uniform(self.embedding)
    self.graph_encoder = GraphEncoder(hparams).to(self.hparams.device)


  def forward(self, token, adj):
    self.graph_embedding = self.graph_encoder(self.embedding, adj)
    start_emb = self.graph_embedding[token[:, 0]]
    end_emb = self.graph_embedding[token[:, -1]]
    cat_emb = torch.cat([start_emb, end_emb], 1)
    pred = self.linear(cat_emb)
    return pred




class ETAModel(Module):
  def __init__(self, hparams):
    super(ETAModel, self).__init__()
    self.hparams = hparams
    self.BiGRU = GRU(self.hparams).to(self.hparams.device)

    self.linear = torch.nn.Linear(hparams.gru_hidden_size * 2, 1)

    self.embedding_size = hparams.node_dims
    self.vocab_size = hparams.vocab_size
    self.embedding = Parameter(torch.FloatTensor(hparams.node_num, hparams.node_dims))
    torch.nn.init.xavier_uniform(self.embedding)
    self.graph_encoder = GraphEncoder(hparams).to(self.hparams.device)


  def forward(self, token, adj):
    self.graph_embedding = self.graph_encoder(self.embedding, adj)
    output, _  = self.BiGRU(token, self.graph_embedding)  #batch_size * max_len * hidden_dims
    output = output.view(-1, self.hparams.batch_size, 2, self.hparams.gru_hidden_size)
    output = torch.cat([output[:, :, 0, :], output[:, :, 1, :]], 2)
    max_out, _ = torch.max(output, 0) #batch_size * hidden_dims
    pred = self.linear(max_out)
    return pred



class GRU(nn.Module):
  def __init__(self, hparams):
    super(GRU, self).__init__()
    self.device = hparams.device
    self.emb_size = hparams.node_dims
    self.node_num = hparams.node_num
    self.hidden_size = hparams.gru_hidden_size
    self.batch_size = hparams.batch_size
    self.is_bigru = hparams.is_bigru
    self.state_num = hparams.state_num
    self.gru = nn.GRU(self.emb_size, self.hidden_size, bidirectional=self.is_bigru)
    self.out = nn.Linear(self.hidden_size, self.node_num)
  def forward(self, token, graph_embedding): 
    token_emb = graph_embedding[token, :]
    output = self.gru(token_emb.transpose(1, 0), self.initHidden())
    return output

  def initHidden(self):
    return torch.zeros(self.state_num, self.batch_size, self.hidden_size, device=self.device)

class GraphEncoder(Module):
  def __init__(self, hparams):
    super(GraphEncoder, self).__init__()
    self.hparams = hparams
    self.node_num = self.hparams.node_num
    self.cmt_num = self.hparams.cmt_num
    self.cmt_weight = Parameter(torch.FloatTensor(self.node_num, self.cmt_num))
    torch.nn.init.xavier_uniform(self.cmt_weight)

    self.gnn_layers = hparams.gnn_layers
    self.graph_encoders = []
    for i in range(self.gnn_layers):
      if hparams.use_cn_gnn:
        graph_encoder = CommunityNodeGNN(hparams).to(self.hparams.device)
      else:
        graph_encoder = PlainGCN(hparams).to(self.hparams.device)
      self.graph_encoders.append(graph_encoder)
  def forward(self, inputs, adj):  #one layer CN-GNN
    for graph_encoder in self.graph_encoders:
      inputs = graph_encoder(inputs, adj, self.cmt_weight)
    return inputs


class PlainGCN(Module):
  def __init__(self, hparams):
    super(PlainGCN, self).__init__()
    self.hparams = hparams

#    self.plain_conv = GraphConvolution(
#        in_features=self.hparams.node_dims,
#        out_features=self.hparams.node_dims,
#        device=self.hparams.device).to(self.hparams.device)

    self.plain_conv = SPGAT(
        in_features = self.hparams.node_dims,
        out_features = self.hparams.node_dims,
        dropout = self.hparams.dropout,
        alpha = self.hparams.alpha,
        device = self.hparams.device).to(self.hparams.device)


  def forward(self, inputs, adj, cmt_weight):  #one layer CN-GNN
    node_out = F.relu(self.plain_conv(inputs.unsqueeze(0), adj.unsqueeze(0).float())).squeeze()
    return node_out


class CommunityNodeGNN(Module):
  def __init__(self, hparams):
    super(CommunityNodeGNN, self).__init__()
    self.hparams = hparams

    self.plain_node_conv = GraphConvolution(
        in_features=self.hparams.node_dims + self.hparams.cmt_dims,
        out_features=self.hparams.node_dims,
        device=self.hparams.device).to(self.hparams.device)

    self.plain_cmt_conv = GraphConvolution(
        in_features=self.hparams.cmt_dims,
        out_features=self.hparams.cmt_dims,
        device=self.hparams.device).to(self.hparams.device)

  def forward(self, inputs, adj, cmt_weight):  #one layer CN-GNN
    cmt_weight_softmax = F.softmax(cmt_weight, dim=-1)
    cmt_emb, cmt_adj, _, _ = get_cmt(inputs, adj.float(), cmt_weight_softmax)
    cmt_emb = self.cmt_forward(cmt_emb, cmt_adj)
    inputs = self.node_forward(inputs, adj.float(), cmt_emb, cmt_weight_softmax)
    return inputs

  def node_forward(self, inputs, adj, cmt_emb, cmt_weight_softmax, embedding_mask=None):
#    node_cmt_emb = 
    node2cmt = torch.argmax(cmt_weight_softmax, -1) # node_num * 1
    cmt_emb = cmt_emb.squeeze()
    inputs_cmt_emb = cmt_emb[node2cmt, :]
    inputs_cat = torch.cat([inputs, inputs_cmt_emb], -1)
    node_out = F.relu(self.plain_node_conv(inputs_cat.unsqueeze(0), adj.unsqueeze(0))).squeeze()
    return node_out

  def cmt_forward(self, cmt_emb, cmt_adj, embedding_mask=None):  # apply plain gcn on cmt level 
    cmt_out = F.relu(self.plain_cmt_conv(cmt_emb, cmt_adj)).squeeze()
    if embedding_mask is not None:
      cmt_out = cmt_out * embedding_mask
    return cmt_out


















