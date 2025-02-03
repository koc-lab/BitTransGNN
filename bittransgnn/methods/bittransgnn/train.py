from pathlib import Path

import yaml

from .run_method import run_bittransgnn
from logger import Logger
from utils import get_model_type, get_train_state, set_seed

config_path = Path(__file__).parent.joinpath("./configs")
config_file_path = config_path.joinpath("./train_config.yaml")
with open(config_file_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
exp_configs = config["experiment_configs"]
model_configs = config["model_configs"]
parameters = config["parameters"]
log_configs = config["log_configs"]
set_seed(exp_configs["seed"])
model_checkpoint, ckpt_dir, best_metrics = run_bittransgnn(config)
log_ckpt, save_ckpt = exp_configs["log_ckpt"], exp_configs["save_ckpt"]
model_configs["model_type"] = get_model_type(model_configs["quantize_bert"], model_configs["quantize_gcn"])
model_configs["train_state"] = get_train_state(parameters["joint_training"])
logger = Logger(log_configs["comet"], log_configs["pandas_df"], log_configs["wandb"], log_ckpt, save_ckpt,
                api_key=log_configs["api_key"], workspace=log_configs["workspace"])
logger.log(config, best_metrics, 
            model_name="bittransgnn", project_name="bittransgnn", 
            model_checkpoint=model_checkpoint, ckpt_dir=ckpt_dir)
