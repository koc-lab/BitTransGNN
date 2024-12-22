from pathlib import Path

import yaml

from .run_method import run_bitbertgcn_kd_for_inference
from logger import Logger
from utils import get_model_type, get_train_state, set_seed

config_path = Path(__file__).parent.joinpath("./configs")
config_file_path = config_path.joinpath("./inference_config.yaml")
with open(config_file_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
exp_configs = config["experiment_configs"]
model_configs = config["model_configs"]
parameters = config["parameters"]
log_configs = config["log_configs"]
set_seed(exp_configs["seed"])
best_metrics, ckpt_dir = run_bitbertgcn_kd_for_inference(config)
model_configs["teacher_model_type"] = get_model_type(model_configs["quantize_teacher_bert"], model_configs["quantize_gcn"])
model_configs["teacher_train_state"] = get_train_state(parameters["joint_training"])
log_ckpt, save_ckpt = exp_configs["log_ckpt"], exp_configs["save_ckpt"]
logger = Logger(log_configs["comet"], log_configs["pandas_df"], log_configs["wandb"], log_ckpt, save_ckpt)
logger.log(config, best_metrics, 
            model_name="bitbertgcn_kd_inference", project_name="bitbertgcn_kd_inference",
            ckpt_dir=ckpt_dir)

