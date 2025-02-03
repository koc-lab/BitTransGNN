from pathlib import Path

import yaml

from .run_method import run_bittransformer
from logger import Logger
from utils import set_seed

config_path = Path(__file__).parent.joinpath("./configs")
config_file_path = config_path.joinpath("./train_config.yaml")
with open(config_file_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
exp_configs = config["experiment_configs"]
model_configs = config["model_configs"]
parameters = config["parameters"]
log_configs = config["log_configs"]
parameters_log = {par_name: parameters[par_name] for par_name in parameters.keys()}
for dataset_name in exp_configs["dataset_name"]:
    for seed in exp_configs["seed"]:
        for num_states in model_configs["num_states"]:
            if num_states == 0:
                quantize_bert = False
            else:
                quantize_bert = True
            set_seed(seed)
            bert_pre_model = exp_configs["bert_pre_model"]
            parameters_log["bert_pre_model"] = bert_pre_model
            parameters_log["dataset_name"], parameters_log["seed"], parameters_log["num_states"] = dataset_name, seed, num_states
            parameters_log["quantize_bert"] = quantize_bert
            iter_exp_configs = {name: exp_configs[name] for name in exp_configs.keys()}
            iter_model_configs = {name: model_configs[name] for name in model_configs.keys()}
            iter_exp_configs["dataset_name"] = dataset_name
            iter_exp_configs["bert_pre_model"] = bert_pre_model
            iter_exp_configs["seed"] = seed
            iter_model_configs["num_states"] = num_states
            iter_model_configs["quantize_bert"] = quantize_bert
            iter_config = {"experiment_configs": iter_exp_configs, "model_configs": iter_model_configs, "parameters": parameters_log}
            model_checkpoint, ckpt_dir, best_metrics, logits = run_bittransformer(iter_config)
            log_ckpt, save_ckpt, save_logits = exp_configs["log_ckpt"], exp_configs["save_ckpt"], exp_configs["save_logits"]
            logger = Logger(log_configs["comet"], log_configs["pandas_df"], log_configs["wandb"], log_ckpt, save_ckpt, save_logits,
                            api_key=log_configs["api_key"], workspace=log_configs["workspace"])
            logger.log(iter_config, best_metrics, logits,
                        model_name="bittrans_train", project_name="bittrans_train", 
                        model_checkpoint=model_checkpoint, ckpt_dir=ckpt_dir, parameters=parameters_log)
