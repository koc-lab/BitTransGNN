import os

import torch

from utils import get_quant_name, get_train_state, get_model_type, set_bittransgnn_ckpt_dir, get_pretrained_transformer_ckpt, load_bittransgnn_for_inference
from data.loader.dataloaders import GraphDataObject, TextDataObject
from models import BitTransGNN
from trainers import BitTransGNNTrainer
from inference_engines import BitTransGNNInference
from quantization.binarize_model import quantize_bittransgnn_architecture, quantize_bittransgnn_for_inference

def run_bittransgnn(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_bert, quantize_gcn, quantize_embeddings, quantize_attention, num_bits_act, num_states, gcn_num_states = model_configs["max_length"], model_configs["quantize_bert"], model_configs["quantize_gcn"], \
        model_configs["quantize_embeddings"], model_configs["quantize_attention"], model_configs["num_bits_act"], model_configs["num_states"], model_configs["gcn_num_states"]
    dataset_name, bert_pre_model, device, report_time, checkpoint_dir, nb_epochs, patience = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"], exp_configs["checkpoint_dir"], exp_configs["nb_epochs"], exp_configs["patience"]
    bert_lr, gcn_lr, batch_size, joint_training, bert_quant_type, dropout, n_hidden = parameters["bert_lr"], parameters["gcn_lr"], parameters["batch_size"], parameters["joint_training"], parameters["bert_quant_type"], parameters["dropout"], parameters["graph_hidden_size"]
    lmbd, gcn_layers = parameters["lmbd"], parameters["gcn_layers"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    save_ckpt = exp_configs["save_ckpt"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    inference_type = exp_configs["inference_type"]
    eval_test, eval_test_every_n_epochs = exp_configs["eval_test"], exp_configs["eval_test_every_n_epochs"]
    seed = exp_configs["seed"]
    adj_type=config["model_configs"].get("adj_type", None)

    if inference_type == "inductive":
        inductive = True
        train_only = True
    else:
        inductive = False
        train_only = False

    quant_name = get_quant_name(num_states)
    train_state = get_train_state(joint_training)
    model_type = get_model_type(quantize_bert, quantize_gcn)
    
    ckpt_dir_dict = set_bittransgnn_ckpt_dir(checkpoint_dir, 
                                            model_type, 
                                            quantize_bert, quantize_embeddings, quantize_attention, bert_quant_type, 
                                            bert_pre_model, 
                                            train_state, quant_name, 
                                            dataset_name, 
                                            num_bits_act, 
                                            adj_type,
                                            inference_type=inference_type)
    if save_ckpt:
        print("model_ckpt_dir", ckpt_dir_dict["model_ckpt_dir"])
        os.makedirs(ckpt_dir_dict["model_ckpt_dir"], exist_ok=True)

    ckpt = get_pretrained_transformer_ckpt(quantize_bert, quantize_embeddings, quantize_attention, bert_pre_model, quant_name, dataset_name, bert_quant_type, num_bits_act,
                                    local_load, manual_load_ckpt, experiment_load_name, experiment_load_key, api_key, workspace, student=False)

    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset_name)
    #idx_loaders, nb_node, nb_train, nb_val, nb_test, nb_word, nb_class = get_dataloaders_bertgcn(dataset_name, batch_size, features, y_train, train_mask, val_mask, test_mask)
    
    graph_data = GraphDataObject(dataset_name, batch_size, adj_type=adj_type, seed=seed, train_only=train_only)
    
    nb_class = graph_data.nb_class
    regression = dataset_name == "stsb"
    model = BitTransGNN(pretrained_model=bert_pre_model, joint_training=joint_training, 
                         quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                         nb_class=nb_class, lmbd=lmbd, gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                         regression=regression)
    graph_data.set_transformer_data(model, max_length)
    graph_data.set_graph_data(model)

    model = quantize_bittransgnn_architecture(model, ckpt, 
                                          quantize_bert, bert_quant_type, 
                                          quantize_gcn, num_states, 
                                          joint_training, lmbd, 
                                          quantize_embeddings, num_bits_act, quantize_attention)
    
    model = model.to(device)

    lr = gcn_lr
    if joint_training:
        optimizer = torch.optim.Adam([
                {'params': model.bert_model.parameters(), 'lr': bert_lr},
                {'params': model.classifier.parameters(), 'lr': bert_lr},
                {'params': model.gcn.parameters(), 'lr': gcn_lr},
            ], lr=lr
        )
    else:
        optimizer = torch.optim.Adam([
                #{'params': model.classifier.parameters(), 'lr': bert_lr},
                {'params': model.gcn.parameters(), 'lr': gcn_lr},
            ], lr=lr
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[nb_epochs//2], gamma=0.1)

    trainer = BitTransGNNTrainer(model, dataset_name, optimizer, scheduler, graph_data, joint_training, device, batch_size, inductive, eval_test, eval_test_every_n_epochs)
    model_checkpoint, best_metrics, logits = trainer.run(nb_epochs, patience, report_time)
    return model_checkpoint, ckpt_dir_dict, best_metrics, logits

def run_bittransgnn_for_inference(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_bert, quantize_gcn, quantize_embeddings, quantize_attention, num_bits_act, num_states, gcn_num_states = model_configs["max_length"], model_configs["quantize_bert"], model_configs["quantize_gcn"], \
        model_configs["quantize_embeddings"], model_configs["quantize_attention"], model_configs["num_bits_act"], model_configs["num_states"], model_configs["gcn_num_states"]
    dataset_name, bert_pre_model, device, report_time = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"]
    batch_size, joint_training, dropout, n_hidden = parameters["batch_size"], parameters["joint_training"], parameters["dropout"], parameters["graph_hidden_size"]
    lmbd, gcn_layers = parameters["lmbd"], parameters["gcn_layers"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    #loaded_bitbert_quant_type = load_configs["loaded_bitbert_quant_type"]
    loaded_bitbert_quant_type = parameters["bert_quant_type"]
    checkpoint_dir = exp_configs["checkpoint_dir"]
    inference_type = exp_configs["inference_type"]
    seed = exp_configs["seed"]
    adj_type = model_configs["adj_type"]
    linear_backend, embedding_backend, attn_backend = model_configs.get("linear_backend", None), model_configs.get("embedding_backend", None), model_configs.get("attn_backend", None)

    if inference_type == "inductive":
        inductive = True
        train_only = True
    else:
        inductive = False
        train_only = False

    quant_name = get_quant_name(num_states)
    train_state = get_train_state(joint_training)
    model_type = get_model_type(quantize_bert, quantize_gcn)
    
    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset_name)
    #idx_loaders, nb_node, nb_train, nb_val, nb_test, nb_word, nb_class = get_dataloaders_bertgcn(dataset_name, batch_size, features, y_train, train_mask, val_mask, test_mask)

    graph_data = GraphDataObject(dataset_name, batch_size, adj_type=adj_type, seed=seed, train_only=train_only)
    nb_class = graph_data.nb_class

    regression = dataset_name == "stsb"

    ckpt = load_bittransgnn_for_inference(model_type, bert_pre_model,
                                         quantize_bert, quantize_embeddings, quantize_attention,
                                         loaded_bitbert_quant_type, 
                                         train_state, quant_name, 
                                         dataset_name,
                                         adj_type, num_bits_act,
                                         local_load, manual_load_ckpt, 
                                         workspace, api_key, 
                                         experiment_load_name, experiment_load_key,
                                         bittransgnn_inference_type=inference_type)

    if lmbd == -1:
        lmbd = ckpt["lmbd"]
    else:
        assert (ckpt["lmbd"] == lmbd)
    print(f"lmbd: {lmbd}")

    model = BitTransGNN(pretrained_model=bert_pre_model, joint_training=joint_training, 
                         quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                         nb_class=nb_class, lmbd=lmbd, gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                         regression=regression)
    graph_data.set_transformer_data(model, max_length)
    graph_data.set_graph_data(model)

    log_dir_dict = set_bittransgnn_ckpt_dir(checkpoint_dir, model_type, quantize_bert, quantize_embeddings, quantize_attention, loaded_bitbert_quant_type, bert_pre_model, train_state, quant_name, dataset_name, num_bits_act, adj_type, inference=True, inference_type=inference_type)

    model = quantize_bittransgnn_for_inference(model, ckpt, 
                                              joint_training, 
                                              quantize_bert, num_states, loaded_bitbert_quant_type, quantize_embeddings, num_bits_act=num_bits_act, quantize_attention=quantize_attention, 
                                              linear_backend=linear_backend, embedding_backend=embedding_backend, attn_backend=attn_backend)

    model = model.to(device)

    if "cls_feats" in ckpt.keys():
        ext_cls_feats = ckpt["cls_feats"]
        print("The provided ext_cls_feats tensor from the model checkpoint will be used to initialize cls_feats.")
    else:
        ext_cls_feats = None
        print("No cls_feats tensor uploaded, the loaded BERT model instance will be used to initialize cls_feats.")

    inference_engine = BitTransGNNInference(model, dataset_name, graph_data, joint_training, device, ext_cls_feats, batch_size, inductive)
    inference_metrics, logits = inference_engine.run(report_time)
    return inference_metrics, log_dir_dict, logits
