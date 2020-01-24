import torch
from utils import *
from conf import beijing_hparams
hparams = dict_to_object(beijing_hparams)
adj, features = load_gae_data(hparams)
gae = torch.load("/data/wuning/NTLR/beijing/model/gae.model_300")

