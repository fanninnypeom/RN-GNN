beijing_hparams = {
    "hidden_dims" : 608,
    "node_dims" : 512,
    "node_num" : 16000,
    "cmt_num" : 200,
    "cmt_dims" : 256,
    "struct_cmt_num" : 300,
    "struct_cmt_dims" : 608,
    "fnc_cmt_num" : 30,
    "fnc_cmt_dims" : 256,
    "train_cmt_set" : "/data/wuning/RN-GNN/beijing/train_cmt_set",
    "train_loc_set" : "/data/wuning/NTLR/beijing/train_loc_set",
    "train_time_set" : "/data/wuning/NTLR/beijing/train_time_set_eta",
    "adj" : "/data/wuning/map-matching/allGraph",
    "node_features" : "/data/wuning/RN-GNN/beijing/node_features",
    "spectral_label" : "/data/wuning/RN-GNN/beijing/spectral_label",
    "struct_assign" : "/data/wuning/RN-GNN/beijing/spectral_label",
    "fnc_assign" : "/data/wuning/RN-GNN/beijing/fnc_assign",

    "gru_dims": 512,
    "gru_layers" : 1,
    "is_bigru" : True,
    "state_num" : 2,
    "vocab_size" : 16000,
    "batch_size" : 100,
    "device": 2,
    "use_cn_gnn" : False,
    "gnn_layers" : 1,
    "eta_epoch" : 20,
    "gae_epoch" : 1000,
    "eta_learning_rate" : 1e-4,
    "gae_learning_rate" : 5e-4,
    "g2s_learning_rate" : 1e-4,
    "lp_learning_rate" : 1e-4,
    "loc_pred_gnn_layer" : 1,

    "alpha":0.2,
    "dropout":0.6,

    "lane_num":6,
    "length_num":220,
    "type_num":20,
    "lane_dims":32,
    "length_dims":32,
    "type_dims":32,

    "clip":0.1,
    "g2s_clip":1.0,
    "lp_clip":1.0

    }


