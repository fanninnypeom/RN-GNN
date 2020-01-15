beijing_hparams = {
    "node_dims" : 256,
    "node_num" : 16000,
    "cmt_num" : 200,
    "cmt_dims" : 256,

    "train_loc_set" : "/data/wuning/NTLR/beijing/train_loc_set",
    "train_time_set" : "/data/wuning/NTLR/beijing/train_time_set_eta",
    "adj" : "/data/wuning/map-matching/allGraph",

    "gru_hidden_size": 512,
    "gru_layers" : 1,
    "is_bigru" : True,
    "state_num" : 2,
    "vocab_size" : 16000,
    "batch_size" : 100,
    "device": 0,
    "use_cn_gnn" : False,
    "gnn_layers" : 1,
    "eta_epoch" : 20,
    "eta_learning_rate" : 1e-4,

    "alpha":0.2,
    "dropout":0.6
    }


