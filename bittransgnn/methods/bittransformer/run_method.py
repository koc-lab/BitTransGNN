import os

import torch
from torch.optim import lr_scheduler

from utils import get_quant_name, set_bitbert_ckpt_dir, get_pretrained_bert_ckpt
from quantization.binarize_model import quantize_bitbert_for_inference
from data.loader.dataloaders import TextDataObject
from models import BertClassifier
from trainers import BitBERTTrainer
from inference_engines import BitBERTInference

def run_bitbert(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    max_length, quantize_bert, quantize_embeddings, num_bits_act, num_states = model_configs["max_length"], model_configs["quantize_bert"], \
        model_configs["quantize_embeddings"], model_configs["num_bits_act"], model_configs["num_states"]
    dataset_name, bert_pre_model, device, report_time, checkpoint_dir, nb_epochs, patience = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"], exp_configs["checkpoint_dir"], exp_configs["nb_epochs"], exp_configs["patience"]
    bert_lr, batch_size = parameters["bert_lr"], parameters["batch_size"]
    save_ckpt = exp_configs["save_ckpt"]
    eval_test, eval_test_every_n_epochs = exp_configs["eval_test"], exp_configs["eval_test_every_n_epochs"]
    seed = exp_configs["seed"]

    quant_name = get_quant_name(num_states)

    ckpt_dir_dict = set_bitbert_ckpt_dir(checkpoint_dir, quantize_bert, quantize_embeddings, bert_pre_model, quant_name, dataset_name, num_bits_act)
    if save_ckpt:
        print("model_ckpt_dir", ckpt_dir_dict["model_ckpt_dir"])
        os.makedirs(ckpt_dir_dict["model_ckpt_dir"], exist_ok=True)

    text_data = TextDataObject(dataset_name, batch_size, seed)
    nb_class = text_data.nb_class
    
    regression = dataset_name == "stsb"
    model = BertClassifier(pretrained_model=bert_pre_model, nb_class=nb_class, quantize=quantize_bert, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act, regression=regression)
    model = model.to(device)
    text_data.set_dataloaders_bert(model, max_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=bert_lr)
    milestone = nb_epochs//2
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=0.1)

    trainer = BitBERTTrainer(model, dataset_name, optimizer, scheduler, text_data, device, eval_test, eval_test_every_n_epochs)
    model_checkpoint, best_metrics = trainer.run(nb_epochs, patience, report_time, ckpt_dir_dict["model_ckpt_dir"])
    return model_checkpoint, ckpt_dir_dict, best_metrics

def run_bitbert_for_inference(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_bert, quantize_embeddings, num_bits_act, num_states = model_configs["max_length"], model_configs["quantize_bert"], \
        model_configs["quantize_embeddings"], model_configs["num_bits_act"], model_configs["num_states"]
    dataset_name, bert_pre_model, device, report_time = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    batch_size = parameters["batch_size"]
    loaded_bitbert_quant_type = parameters["bert_quant_type"]
    checkpoint_dir = exp_configs["checkpoint_dir"]

    quant_name = get_quant_name(num_states)

    text_data = TextDataObject(dataset_name, batch_size)
    nb_class = text_data.nb_class

    ckpt = get_pretrained_bert_ckpt(quantize_bert, quantize_embeddings, bert_pre_model, quant_name, dataset_name, loaded_bitbert_quant_type, 
                                    num_bits_act, 
                                    local_load, manual_load_ckpt, 
                                    experiment_load_name, experiment_load_key, 
                                    api_key, workspace,
                                    student=False)
    
    log_dir_dict = set_bitbert_ckpt_dir(checkpoint_dir, quantize_bert, quantize_embeddings, bert_pre_model, quant_name, dataset_name, num_bits_act, inference=True)

    regression = dataset_name == "stsb"
    model = BertClassifier(pretrained_model=bert_pre_model, nb_class=nb_class, quantize=quantize_bert, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act, regression=regression)
    joint_training = False #not applicable but set to False since quantize_bitbert_for_inference() is used for inference with both BitBERT and BitBERTGCN models
    model = quantize_bitbert_for_inference(model, ckpt, joint_training, quantize_bert, num_states, loaded_bitbert_quant_type, quantize_embeddings, num_bits_act)
    model = model.to(device)
    text_data.set_dataloaders_bert(model, max_length)

    inference_engine = BitBERTInference(model, dataset_name, text_data, device)
    inference_metrics = inference_engine.run(report_time)
    return inference_metrics, log_dir_dict
