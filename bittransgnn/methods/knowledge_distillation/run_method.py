import os

import torch

from utils import get_quant_name, get_train_state, get_model_type, set_bittransgnn_kd_ckpt_dir, get_pretrained_bert_ckpt, get_pretrained_teacher_bittransgnn_ckpt, load_bittransgnn_kd_student_for_inference
from data.loader.dataloaders import GraphDataObject, TextDataObject
from models import BitTransGNN, BitTransformerStudent, BitTransformer
from trainers import BitTransGNNKDTrainer
from inference_engines import BitTransformerInference
from quantization.binarize_model import quantize_teacher_architecture, quantize_student_architecture

def run_bittransgnn_kd(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_bert, quantize_teacher_bert, quantize_gcn, quantize_embeddings, quantize_teacher_embeddings, num_bits_act, teacher_num_states, student_num_states, gcn_num_states = model_configs["max_length"], model_configs["quantize_bert"], model_configs["quantize_teacher_bert"], model_configs["quantize_gcn"], \
        model_configs["quantize_embeddings"], model_configs["quantize_teacher_embeddings"], model_configs["num_bits_act"], model_configs["teacher_num_states"], model_configs["student_num_states"], model_configs["gcn_num_states"]
    dataset_name, bert_pre_model, device, report_time, checkpoint_dir, nb_epochs, patience = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"], exp_configs["checkpoint_dir"], exp_configs["nb_epochs"], exp_configs["patience"]
    bert_lr, gcn_lr, batch_size, joint_training, student_bert_quant_type, teacher_bert_quant_type, dropout, n_hidden = parameters["bert_lr"], parameters["gcn_lr"], parameters["batch_size"], parameters["joint_training"], parameters["student_bert_quant_type"], parameters["teacher_bert_quant_type"], parameters["dropout"], parameters["graph_hidden_size"]
    teacher_lmbd, gcn_layers = parameters["teacher_lmbd"], parameters["gcn_layers"]
    save_ckpt = exp_configs["save_ckpt"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    distillation_type = exp_configs["distillation_type"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    alpha_d, temperature = parameters["alpha_d"], parameters["temperature"]
    inference_type = exp_configs["inference_type"]
    eval_test, eval_test_every_n_epochs = exp_configs["eval_test"], exp_configs["eval_test_every_n_epochs"]

    if inference_type == "inductive":
        inductive = True
        train_only = True
    else:
        inductive = False
        train_only = False

    teacher_quant_name = get_quant_name(teacher_num_states)
    student_quant_name = get_quant_name(student_num_states)
    train_state = get_train_state(joint_training)
    model_type = get_model_type(quantize_teacher_bert, quantize_gcn)

    ckpt_dir_dict = set_bittransgnn_kd_ckpt_dir(checkpoint_dir, model_type, quantize_bert, quantize_embeddings, 
                                               student_bert_quant_type, bert_pre_model, train_state, student_quant_name, 
                                               dataset_name, num_bits_act, inference_type=inference_type)
    if save_ckpt:
        print("model_ckpt_dir", ckpt_dir_dict["model_ckpt_dir"])
        os.makedirs(ckpt_dir_dict["model_ckpt_dir"], exist_ok=True)

    student_ckpt = get_pretrained_bert_ckpt(quantize_bert, quantize_embeddings, bert_pre_model, student_quant_name, dataset_name, 
                                            student_bert_quant_type, num_bits_act, local_load, 
                                            manual_load_ckpt, experiment_load_name, experiment_load_key, api_key, workspace, 
                                            student=True)
    print("Transformer type: ", bert_pre_model)

    teacher_ckpt = get_pretrained_teacher_bittransgnn_ckpt(model_type, bert_pre_model, distillation_type,
                                                          quantize_teacher_bert, quantize_teacher_embeddings, 
                                                          teacher_bert_quant_type, 
                                                          train_state, teacher_quant_name, 
                                                          dataset_name, 
                                                          student_ckpt, local_load, manual_load_ckpt, 
                                                          workspace, api_key, experiment_load_name, experiment_load_key,
                                                          bittransgnn_inference_type=inference_type)
    print("Number of states of BitTransGNN model parameters: ", teacher_num_states)

    graph_data = GraphDataObject(dataset_name, batch_size, train_only)
    nb_class = graph_data.nb_class

    regression = dataset_name == "stsb"
    teacher_model = BitTransGNN(pretrained_model=bert_pre_model, joint_training=joint_training, 
                                 quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                                 nb_class=nb_class, lmbd=teacher_ckpt["lmbd"], gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                                 regression=regression)
    graph_data.set_transformer_data(teacher_model, max_length)
    graph_data.set_graph_data(teacher_model)
    
    teacher_model = quantize_teacher_architecture(teacher_model, teacher_ckpt, 
                                                  joint_training, 
                                                  quantize_teacher_bert, teacher_bert_quant_type, teacher_num_states, quantize_teacher_embeddings, 
                                                  num_bits_act)
    
    if teacher_lmbd is not None:
        assert (teacher_lmbd == teacher_ckpt["lmbd"])

    student_model = BitTransformerStudent(pretrained_model=bert_pre_model, nb_class=nb_class, regression=regression)

    student_model = quantize_student_architecture(student_model, student_ckpt, quantize_bert, student_bert_quant_type, student_num_states, 
                                                  quantize_embeddings, num_bits_act)

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    lr = gcn_lr
    teacher_optimizer = torch.optim.Adam([
            {'params': teacher_model.bert_model.parameters(), 'lr': bert_lr},
            {'params': teacher_model.classifier.parameters(), 'lr': bert_lr},
            {'params': teacher_model.gcn.parameters(), 'lr': gcn_lr},
        ], lr=lr
    )
    student_optimizer = torch.optim.Adam([
            {'params': student_model.bert_model.parameters(), 'lr': bert_lr},
            {'params': student_model.classifier.parameters(), 'lr': bert_lr},
        ], lr=bert_lr
    )
    teacher_scheduler = torch.optim.lr_scheduler.MultiStepLR(teacher_optimizer, milestones=[nb_epochs//2], gamma=0.1)
    student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, milestones=[nb_epochs//2], gamma=0.1)

    trainer = BitTransGNNKDTrainer(teacher_model, student_model, dataset_name,
                                  student_optimizer, student_scheduler, 
                                  graph_data,
                                  alpha_d, temperature, 
                                  device, 
                                  batch_size,
                                  distillation_type, 
                                  teacher_optimizer, teacher_scheduler,
                                  inductive,
                                  eval_test, eval_test_every_n_epochs)
    model_checkpoint, best_metrics = trainer.run(nb_epochs, patience, report_time)
    return model_checkpoint, ckpt_dir_dict, best_metrics    

def run_bittransgnn_kd_for_inference(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_bert, quantize_teacher_bert, quantize_gcn, quantize_embeddings, num_bits_act, student_num_states = model_configs["max_length"], model_configs["quantize_bert"], model_configs["quantize_teacher_bert"], model_configs["quantize_gcn"], model_configs["quantize_embeddings"], model_configs["num_bits_act"], model_configs["student_num_states"]
    dataset_name, bert_pre_model, device, report_time = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"]
    batch_size, joint_training, student_bert_quant_type = parameters["batch_size"], parameters["joint_training"], parameters["student_bert_quant_type"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    checkpoint_dir = exp_configs["checkpoint_dir"]
    inference_type = exp_configs["inference_type"]

    student_quant_name = get_quant_name(student_num_states)
    train_state = get_train_state(joint_training)
    model_type = get_model_type(quantize_teacher_bert, quantize_gcn)

    ckpt_dir_dict = set_bittransgnn_kd_ckpt_dir(checkpoint_dir, model_type, quantize_bert, quantize_embeddings, 
                                               student_bert_quant_type, bert_pre_model, train_state, student_quant_name, 
                                               dataset_name, num_bits_act, inference=True, inference_type=inference_type)

    text_data = TextDataObject(dataset_name, batch_size)
    nb_class = text_data.nb_class

    student_ckpt = load_bittransgnn_kd_student_for_inference(quantize_bert, quantize_embeddings, 
                                                            bert_pre_model, student_quant_name, 
                                                            dataset_name, student_bert_quant_type, 
                                                            num_bits_act, 
                                                            local_load, manual_load_ckpt, 
                                                            experiment_load_name, experiment_load_key, 
                                                            api_key, workspace,
                                                            model_type, train_state,
                                                            bittransgnn_inference_type=inference_type)
        
    regression = dataset_name == "stsb"
    student_model = BitTransformer(pretrained_model=bert_pre_model, nb_class=nb_class, 
                                   quantize=quantize_bert, num_states=student_num_states, 
                                   quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act,
                                   regression=regression)

    student_model = quantize_student_architecture(student_model, student_ckpt, quantize_bert, student_bert_quant_type, student_num_states, quantize_embeddings, num_bits_act)

    student_model = student_model.to(device)
    text_data.set_dataloaders_bert(student_model, max_length)

    inference_engine = BitTransformerInference(student_model, dataset_name, text_data, device)    

    inference_metrics = inference_engine.run(report_time)
    return inference_metrics, ckpt_dir_dict
