from typing import Optional
import os

from utils import get_quant_name, get_train_state, get_model_type

class Logger:
    def __init__(self, comet, pandas_df, wandb, log_ckpt, save_ckpt, api_key=None, workspace=None):
        self.comet = comet
        self.pandas_df = pandas_df
        self.wandb = wandb
        self.log_ckpt = log_ckpt
        self.save_ckpt = save_ckpt
        self.api_key = api_key
        self.workspace = workspace

    def log(self, config, best_metrics, 
            model_name, project_name: Optional[str] = None, 
            model_checkpoint=None, ckpt_dir=None,
            experiment=None, parameters=None):
        if self.comet:
            self.comet_logger(config, best_metrics, model_name, project_name, model_checkpoint, experiment, parameters)
        if self.log_pandas_df:
            self.log_pandas_df(config, best_metrics, model_name, ckpt_dir["result_ckpt_dir"])
        if self.wandb:
            self.wandb_logger(config, best_metrics, model_name, project_name, model_checkpoint)
        if self.save_ckpt:
            self.save_model_local(model_checkpoint, ckpt_dir["model_ckpt_dir"])
    
    def comet_logger(self, config, best_metrics, model_name: Optional[str] = None, project_name: Optional[str] = None, model_checkpoint=None, experiment=None, parameters=None):
        import comet_ml
        if experiment:
            exp = experiment
            if parameters:
                exp.log_parameters(parameters)
            exp.log_parameters(config)
            exp.log_metrics(best_metrics)
            if self.log_ckpt:
                exp.log_model(model_name, model_checkpoint)
        else:
            exp = comet_ml.Experiment(api_key=self.api_key, 
                                    project_name=project_name, #bitbert_train
                                    workspace=self.workspace) 
            if parameters:
                exp.log_parameters(parameters)
            exp.log_parameters(config)
            exp.log_metrics(best_metrics)
            if self.log_ckpt:
                exp.log_model(model_name, model_checkpoint)
            exp.end()

    def log_pandas_df(self, config, best_metrics, model_name, ckpt_dir):
        print(model_name)
        import pandas as pd
        if model_name == "bitbert_train" or model_name == "bitbert_inference":
            log_dict = {"dataset": [], "bert_pre_model": [],
                           "quant_name": [], "bert_quant_type": [],
                           "seed": []}
            for best_metric_name in best_metrics.keys():
                log_dict[best_metric_name] = []
            exp_configs, model_configs, parameters = config["experiment_configs"], config["model_configs"], config["parameters"]
            log_dict["dataset"].append(exp_configs["dataset_name"])
            log_dict["bert_pre_model"].append(exp_configs["bert_pre_model"])
            log_dict["quant_name"].append(get_quant_name(model_configs["num_states"]))
            log_dict["bert_quant_type"].append(parameters["bert_quant_type"])
            log_dict["seed"].append(exp_configs["seed"])
            for best_metric_name in best_metrics.keys():
                log_dict[best_metric_name].append(best_metrics[best_metric_name])

        elif model_name == "bitbertgcn" or model_name == "bitbertgcn_inference" or model_name == "bitbertgcn_direct_seperation":
            log_dict = {"dataset": [], "bert_pre_model": [],
                           "quant_name": [], "bert_quant_type": [],
                           "model_type": [], "train_state": [], "lmbd": [],
                           "seed": []}
            for best_metric_name in best_metrics.keys():
                log_dict[best_metric_name] = []
            exp_configs, model_configs, parameters = config["experiment_configs"], config["model_configs"], config["parameters"]
            log_dict["dataset"].append(exp_configs["dataset_name"])
            log_dict["bert_pre_model"].append(exp_configs["bert_pre_model"])
            log_dict["quant_name"].append(get_quant_name(model_configs["num_states"]))
            log_dict["model_type"].append(get_model_type(model_configs["quantize_bert"], model_configs["quantize_gcn"]))
            log_dict["train_state"].append(get_train_state(parameters["joint_training"]))
            log_dict["bert_quant_type"].append(parameters["bert_quant_type"])
            log_dict["lmbd"].append(parameters["lmbd"])
            log_dict["seed"].append(exp_configs["seed"])
            for best_metric_name in best_metrics.keys():
                log_dict[best_metric_name].append(best_metrics[best_metric_name])


        elif model_name == "bitbertgcn_kd" or model_name == "bitbertgcn_kd_inference":
            log_dict = {"dataset": [], "bert_pre_model": [],
                           "teacher_quant_name": [], "teacher_bert_quant_type": [],
                           "student_quant_name": [], "student_bert_quant_type": [],
                           "teacher_model_type": [], "teacher_train_state": [], "teacher_lmbd": [],
                           "alpha_d": [], "temperature": [],
                           "seed": []}
            for best_metric_name in best_metrics.keys():
                log_dict[best_metric_name] = []
            exp_configs, model_configs, parameters = config["experiment_configs"], config["model_configs"], config["parameters"]
            log_dict["dataset"].append(exp_configs["dataset_name"])
            log_dict["bert_pre_model"].append(exp_configs["bert_pre_model"])
            log_dict["teacher_quant_name"].append(get_quant_name(model_configs["teacher_num_states"]))
            log_dict["student_quant_name"].append(get_quant_name(model_configs["student_num_states"]))
            log_dict["teacher_model_type"].append(get_model_type(model_configs["quantize_teacher_bert"], model_configs["quantize_gcn"]))
            log_dict["teacher_train_state"].append(get_train_state(parameters["joint_training"]))
            log_dict["teacher_bert_quant_type"].append(parameters["teacher_bert_quant_type"])
            log_dict["student_bert_quant_type"].append(parameters["student_bert_quant_type"])
            log_dict["teacher_lmbd"].append(parameters["teacher_lmbd"])
            log_dict["alpha_d"].append(parameters["alpha_d"])
            log_dict["temperature"].append(parameters["temperature"])
            log_dict["seed"].append(exp_configs["seed"])
            for best_metric_name in best_metrics.keys():
                log_dict[best_metric_name].append(best_metrics[best_metric_name])

        print("Logging model performance metrics...")
        print("ckpt_dir", ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        df = pd.DataFrame(log_dict)
        log_file = os.path.join(ckpt_dir, f"{model_name}.csv")
        if os.path.isfile(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, mode='w', header=True, index=False)

    def wandb_logger(self, config, best_metrics, model_name, project_name, model_checkpoint):
        """
        log_model was not implemented for wandb since the experiments mainly used comet
        wandb logger only uploads the metrics and parameters from the experiment
        still, this code is easily adaptable to logging models to wandb platform
        """
        import wandb
        run = wandb.init(project=project_name, config=config)
        wandb.log(best_metrics)

    def save_model_local(self, model_checkpoint, ckpt_dir):
        import torch
        print("Saving model checkpoint to local machine...")
        print("ckpt_dir", ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model_checkpoint, os.path.join(ckpt_dir, "checkpoint.pth"))
