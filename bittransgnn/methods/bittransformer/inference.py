from pathlib import Path

import yaml

from .run_method import run_bittransformer_for_inference
from logger import Logger
from utils import set_seed

config_path = Path(__file__).parent.joinpath("./configs")
config_file_path = config_path.joinpath("./inference_config.yaml")
with open(config_file_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
exp_configs = config["experiment_configs"]
model_configs = config["model_configs"]
parameters = config["parameters"]
log_configs = config["log_configs"]
if isinstance(exp_configs["dataset_name"], list):
    dataset_name_list = exp_configs["dataset_name"]
else:
    dataset_name_list = [exp_configs["dataset_name"]]
if isinstance(exp_configs["bert_pre_model"], list):
    bert_pre_model_list = exp_configs["bert_pre_model"]
else:
    bert_pre_model_list = [exp_configs["bert_pre_model"]]
if isinstance(model_configs["num_states"], list):
    num_states_list = model_configs["num_states"]
else:
    num_states_list = [model_configs["num_states"]]
for dataset_name in dataset_name_list:
    for bert_pre_model in bert_pre_model_list:
        for num_states in num_states_list:
            set_seed(exp_configs["seed"])
            exp_configs["dataset_name"] = dataset_name
            exp_configs["bert_pre_model"] = bert_pre_model
            model_configs["num_states"] = num_states
            config["exp_configs"] = exp_configs
            config["model_configs"] = model_configs
            best_metrics, ckpt_dir = run_bittransformer_for_inference(config)
            log_ckpt, save_ckpt = exp_configs["log_ckpt"], exp_configs["save_ckpt"]
            logger = Logger(log_configs["comet"], log_configs["pandas_df"], log_configs["wandb"], log_ckpt, save_ckpt, 
                            api_key=log_configs["api_key"], workspace=log_configs["workspace"])
            logger.log(config, best_metrics, 
                        model_name="bitbert_inference", project_name="bitbert_inference",
                        ckpt_dir=ckpt_dir)
