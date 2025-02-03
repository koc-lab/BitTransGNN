import os

import torch

from utils import get_quant_name, get_train_state, get_model_type, set_bittransgnn_ckpt_dir, get_pretrained_bert_ckpt, load_bittransgnn_for_inference
from data.loader.dataloaders import GraphDataObject
from models import BitTransGNN
from trainers import BitTransGNNTrainer
from inference_engines import BitTransGNNInference
from quantization.binarize_model import quantize_bertgcn_architecture, quantize_bitbertgcn_for_inference

def run_bittransgnn(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_bert, quantize_gcn, quantize_embeddings, num_bits_act, num_states, gcn_num_states = model_configs["max_length"], model_configs["quantize_bert"], model_configs["quantize_gcn"], \
        model_configs["quantize_embeddings"], model_configs["num_bits_act"], model_configs["num_states"], model_configs["gcn_num_states"]
    dataset_name, bert_pre_model, device, report_time, checkpoint_dir, nb_epochs, patience = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"], exp_configs["checkpoint_dir"], exp_configs["nb_epochs"], exp_configs["patience"]
    bert_lr, gcn_lr, batch_size, joint_training, bert_quant_type, dropout, n_hidden = parameters["bert_lr"], parameters["gcn_lr"], parameters["batch_size"], parameters["joint_training"], parameters["bert_quant_type"], parameters["dropout"], parameters["graph_hidden_size"]
    lmbd, gcn_layers = parameters["lmbd"], parameters["gcn_layers"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    save_ckpt = exp_configs["save_ckpt"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    inference_type = exp_configs["inference_type"]
    eval_test, eval_test_every_n_epochs = exp_configs["eval_test"], exp_configs["eval_test_every_n_epochs"]

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
                                            quantize_bert, quantize_embeddings, bert_quant_type, 
                                            bert_pre_model, 
                                            train_state, quant_name, 
                                            dataset_name, 
                                            num_bits_act, 
                                            inference_type=inference_type)
    if save_ckpt:
        print("model_ckpt_dir", ckpt_dir_dict["model_ckpt_dir"])
        os.makedirs(ckpt_dir_dict["model_ckpt_dir"], exist_ok=True)

    ckpt = get_pretrained_bert_ckpt(quantize_bert, quantize_embeddings, bert_pre_model, quant_name, dataset_name, bert_quant_type, num_bits_act,
                                    local_load, manual_load_ckpt, experiment_load_name, experiment_load_key, api_key, workspace, student=False)
    
    graph_data = GraphDataObject(dataset_name, batch_size, train_only)
    
    nb_class = graph_data.nb_class
    regression = dataset_name == "stsb"
    model = BitTransGNN(pretrained_model=bert_pre_model, joint_training=joint_training,
                         quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                         nb_class=nb_class, lmbd=lmbd, gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                         regression=regression)
    graph_data.set_transformer_data(model, max_length)
    graph_data.set_graph_data(model)

    model = quantize_bertgcn_architecture(model, ckpt, 
                                          quantize_bert, bert_quant_type, 
                                          quantize_gcn, num_states, 
                                          joint_training, lmbd, 
                                          quantize_embeddings, num_bits_act)
    
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
    max_length, quantize_bert, quantize_gcn, quantize_embeddings, num_bits_act, num_states, gcn_num_states = model_configs["max_length"], model_configs["quantize_bert"], model_configs["quantize_gcn"], \
        model_configs["quantize_embeddings"], model_configs["num_bits_act"], model_configs["num_states"], model_configs["gcn_num_states"]
    dataset_name, bert_pre_model, device, report_time = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"]
    batch_size, joint_training, dropout, n_hidden = parameters["batch_size"], parameters["joint_training"], parameters["dropout"], parameters["graph_hidden_size"]
    lmbd, gcn_layers = parameters["lmbd"], parameters["gcn_layers"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    loaded_bittrans_quant_type = parameters["bert_quant_type"]
    checkpoint_dir = exp_configs["checkpoint_dir"]
    inference_type = exp_configs["inference_type"]

    if inference_type == "inductive":
        inductive = True
        train_only = True
    else:
        inductive = False
        train_only = False

    quant_name = get_quant_name(num_states)
    train_state = get_train_state(joint_training)
    model_type = get_model_type(quantize_bert, quantize_gcn)
        
    graph_data = GraphDataObject(dataset_name, batch_size, train_only)
    nb_class = graph_data.nb_class

    regression = dataset_name == "stsb"
    model = BitTransGNN(pretrained_model=bert_pre_model, joint_training=joint_training,
                         quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                         nb_class=nb_class, lmbd=lmbd, gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                         regression=regression)
    graph_data.set_transformer_data(model, max_length)
    graph_data.set_graph_data(model)

    ckpt = load_bittransgnn_for_inference(model_type, bert_pre_model,
                                         quantize_bert, quantize_embeddings, 
                                         loaded_bittrans_quant_type, 
                                         train_state, quant_name, 
                                         dataset_name,
                                         local_load, manual_load_ckpt, 
                                         workspace, api_key, 
                                         experiment_load_name, experiment_load_key,
                                         bittransgnn_inference_type=inference_type)

    log_dir_dict = set_bittransgnn_ckpt_dir(checkpoint_dir, model_type, quantize_bert, quantize_embeddings, loaded_bittrans_quant_type, bert_pre_model, train_state, quant_name, dataset_name, num_bits_act, inference=True, inference_type=inference_type)

    model = quantize_bitbertgcn_for_inference(model, ckpt, 
                                              joint_training, 
                                              quantize_bert, num_states, loaded_bittrans_quant_type, quantize_embeddings, 
                                              num_bits_act)
    
    model = model.to(device)

    inference_engine = BitTransGNNInference(model, dataset_name, graph_data, joint_training, device, batch_size, inductive)
    inference_metrics, logits = inference_engine.run(report_time)
    return inference_metrics, log_dir_dict, logits
