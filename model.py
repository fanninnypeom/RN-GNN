# coding=utf-8
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from gcn_layers import *
import torch.nn as nn
from cmt_gen import get_cmt
import torch.nn.functional as F
import pickle

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

class GraphAutoencoder(Module):
  def __init__(self, hparams):
    super(GraphAutoencoder, self).__init__()
    self.hparams = hparams
    self.enc_in_dims = self.hparams.node_dims + self.hparams.lane_dims + self.hparams.type_dims + self.hparams.length_dims
    self.enc_out_dims = self.hparams.struct_cmt_num
    self.enc_gnn = StructuralGNN(hparams, self.enc_in_dims, self.enc_out_dims).to(self.hparams.device)
    self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(self.hparams.device)
    self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(self.hparams.device)
    self.length_emb_layer = nn.Embedding(hparams.length_num, hparams.length_dims).to(self.hparams.device)
    self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(self.hparams.device)
    self.a = nn.Parameter(torch.zeros(size=(1, 2 * self.enc_in_dims)))
    nn.init.xavier_normal_(self.a.data, gain=1.414)
    self.leakyrelu = nn.LeakyReLU(hparams.alpha)
    self.sigmoid = nn.Sigmoid()
    self.dec_gnn = StructuralDecoder(hparams, self.enc_in_dims, hparams.node_num).to(self.hparams.device)


  def forward(self, main_adj, lane_feature, type_feature, length_feature, node_feature, f_edge):
    node_emb = self.node_emb_layer(node_feature)
    type_emb = self.type_emb_layer(type_feature)
    length_emb = self.length_emb_layer(length_feature)
    lane_emb = self.lane_emb_layer(lane_feature)
    main_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
    struct_emb, struct_adj, main_assign = self.enc_gnn(main_feat, main_adj)  
#    print("struct_emb:", struct_emb.shape, struct_emb)
    res_emb = self.dec_gnn(struct_emb, struct_adj)
#    print("res_emb:", res_emb.shape, res_emb, torch.sum(res_emb, 1))
    t_edge = main_adj._indices().to(self.hparams.device)
    edge = torch.cat([t_edge, f_edge], 1)
    edge_h = torch.cat((res_emb[edge[0, :], :], res_emb[edge[1, :], :]), dim=1).t()
#    print("eeedge h:", edge_h.shape, edge_h, self.a.mm(edge_h).shape, self.a.mm(edge_h))
#    edge_e = self.sigmoid(self.leakyrelu(self.a.mm(edge_h).squeeze()))
    edge_e = self.sigmoid(torch.einsum('ij,ij->i', res_emb[edge[0, :], :], res_emb[edge[1, :], :]))
    edge_label = torch.cat([torch.ones(t_edge.shape[1], dtype=torch.float), torch.zeros(t_edge.shape[1], dtype=torch.float)]).to(self.hparams.device)

    pred_cmt_adj =  torch.mm(struct_emb, struct_emb.t())
    return edge_h, struct_adj, pred_cmt_adj, main_assign, edge_e, edge_label


class Graph2Seq(Module):
  def __init__(self, hparams, lane_feature, type_feature, length_feature, node_feature, main_adj, main_assign):
    super(Graph2Seq, self).__init__() 
    self.hparams = hparams
    self.special_spmm = SpecialSpmm()

    self.fnc_cmt_gat = GraphConvolution(
        in_features = self.hparams.struct_cmt_dims,
        out_features = self.hparams.fnc_cmt_num,
#        dropout = self.hparams.dropout,
#        alpha = self.hparams.alpha,
        device = self.hparams.device).to(self.hparams.device)
    self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(self.hparams.device)
    self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(self.hparams.device)
    self.length_emb_layer = nn.Embedding(hparams.length_num, hparams.length_dims).to(self.hparams.device)
    self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(self.hparams.device)
    node_emb = self.node_emb_layer(node_feature)
    type_emb = self.type_emb_layer(type_feature)
    length_emb = self.length_emb_layer(length_feature)
    lane_emb = self.lane_emb_layer(lane_feature)
    main_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
   
    main_assign = F.softmax(main_assign, 0)
    self.struct_emb = torch.mm(main_assign.t(), main_feat)
    edge = main_adj._indices()#.to(self.hparams.device)
    edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)

    struct_inter = self.special_spmm(edge, edge_e, torch.Size([main_adj.shape[0], main_adj.shape[1]]), main_assign)  #N*N   N*C
    struct_adj = torch.mm(main_assign.t(), struct_inter)
 
    self.fnc_assign = self.fnc_cmt_gat(self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)).squeeze()
    
    self.fnc_assign = F.softmax(self.fnc_assign, 0)

    self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)
    self.fnc_adj = torch.mm(torch.mm(self.fnc_assign.t(), struct_adj), self.fnc_assign)

    self.gru = nn.GRU(hparams.struct_cmt_dims, hparams.struct_cmt_dims)

    self.gru_embedding = nn.Embedding(hparams.struct_cmt_num, hparams.hidden_dims).to(self.hparams.device)
    self.linear = torch.nn.Linear(hparams.hidden_dims * 1, hparams.struct_cmt_num)


  def forward(self, input_bat): #batch_size * length * dims
    init_hidden = torch.zeros(1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device)
    input_emb = self.gru_embedding(input_bat)
    input_state, _ = self.gru(input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]), init_hidden)
    attn_weight = torch.einsum("ijk,lk->ijl", input_state, self.fnc_emb)# cmt_num * cmt_dims  len * batch_size * dims    ->  len * batch_size * cmt_num

#    out_state = torch.cat((input_state, torch.einsum("ijk,kl->ijl", attn_weight, self.fnc_emb)), 2)
#    pred_tra = self.linear(out_state)
    pred_tra = self.linear(input_state)

#    pred_tra = F.softmax(pred_tra, 2)

    return pred_tra

class StructuralDecoder(Module):
  def __init__(self, hparams, in_dims, out_dims):
    super(StructuralDecoder, self).__init__()
    self.hparams = hparams
    self.special_spmm = SpecialSpmm()
    self.cmt_gat_0 = GraphConvolution(
        in_features = in_dims,
        out_features = out_dims,
        device = self.hparams.device).to(self.hparams.device)
    self.softmax = torch.nn.Softmax(dim=-1)
#    self.cmt_gat_1 = GraphConvolution(
#        in_features = self.hparams.node_dims,
#        out_features = out_dims,
#        device = self.hparams.device).to(self.hparams.device)
    
    

  def forward(self, main_feat, main_adj):
    main_assign = self.cmt_gat_0(main_feat.unsqueeze(0), main_adj.unsqueeze(0))
#    print("line 199:", main_inter.shape, main_adj.shape)
#    main_assign = self.cmt_gat_1(main_inter, main_adj.unsqueeze(0))
#    zero_pad_1 = torch.zeros(main_assign.shape[0], brh_assign.shape[1])
#    zero_pad_2 = torch.zeros(brh_assign.shape[0], main_assign.shape[1])
#    top_assign = torch.cat([main_assign, zero_pad_1], 1)
#    buttom_assign = torch.cat([zero_pad_2, brh_assign], 1)

#    struct_assign = torch.cat([top_assign, buttom_assign], 0)

#    all_feat = torch.cat([main_feat, brh_feat], 0)
   
    main_assign = main_assign.to(self.hparams.device)
    main_assign = main_assign.squeeze()
#    print("main assign:", main_assign.shape, main_assign)
    main_assign = F.softmax(main_assign, 0)
#    pickle.dump(main_assign.tolist(), open("struct_assign", "wb"))
    raw_emb = torch.mm(main_assign.t(), main_feat)
    return raw_emb




class StructuralGNN(Module):
  def __init__(self, hparams, in_dims, out_dims):
    super(StructuralGNN, self).__init__()
    self.hparams = hparams
    self.special_spmm = SpecialSpmm()
    self.cmt_gat_0 = SPGAT(
        in_features = in_dims,
        out_features = out_dims,
        dropout = self.hparams.dropout,
        alpha = self.hparams.alpha,
        device = self.hparams.device).to(self.hparams.device)


#    self.cmt_gat_1 = SPGAT(
#        in_features = self.hparams.node_dims,
#        out_features = out_dims,
#        dropout = self.hparams.dropout,
#        alpha = self.hparams.alpha,
#        device = self.hparams.device).to(self.hparams.device)
    
    

  def forward(self, main_feat, main_adj):
    main_assign = self.cmt_gat_0(main_feat, main_adj)
#    main_assign = self.cmt_gat_1(main_inter, main_adj)
#    zero_pad_1 = torch.zeros(main_assign.shape[0], brh_assign.shape[1])
#    zero_pad_2 = torch.zeros(brh_assign.shape[0], main_assign.shape[1])
#    top_assign = torch.cat([main_assign, zero_pad_1], 1)
#    buttom_assign = torch.cat([zero_pad_2, brh_assign], 1)

#    struct_assign = torch.cat([top_assign, buttom_assign], 0)

#    all_feat = torch.cat([main_feat, brh_feat], 0)
   
    main_assign = main_assign.to(self.hparams.device)
    main_assign = F.softmax(main_assign, 0)
#    pickle.dump(main_assign.tolist(), open("struct_assign", "wb"))
    struct_emb = torch.mm(main_assign.t(), main_feat)
    edge = main_adj._indices().to(self.hparams.device)
    edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
    struct_inter = self.special_spmm(edge, edge_e, torch.Size([main_adj.shape[0], main_adj.shape[1]]), main_assign)  #N*N   N*C
    struct_adj = torch.mm(main_assign.t(), struct_inter)
    struct_adj = F.relu(struct_adj - 0.0001)
    return struct_emb, struct_adj, main_assign


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


















