from pathlib import Path

import yaml

from .run_method import run_bittransgnn_for_direct_separation
from logger import Logger
from utils import get_model_type, get_train_state, set_seed

#if __name__ == "__main__":
#if __name__ == "__main__":
config_path = Path(__file__).parent.joinpath("./configs")
#config_file_path = config_path.joinpath("./sweep_config.yaml")
config_file_path = config_path.joinpath("./master_config.yaml")
with open(config_file_path) as file:
    master_config = yaml.load(file, Loader=yaml.FullLoader)
print("---list of configs in master_config---")
print(master_config["configs"])
print('len(master_config["configs"])')
print(len(master_config["configs"]))
count = 0
for config in master_config["configs"]:
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    log_configs = config["log_configs"]
    set_seed(exp_configs["seed"])
    best_metrics, ckpt_dir, logits = run_bittransgnn_for_direct_separation(config)
    model_configs["model_type"] = get_model_type(quantize_bert=True, quantize_gcn=False)
    model_configs["train_state"] = get_train_state(parameters["joint_training"])
    log_ckpt, save_ckpt, save_logits = exp_configs["log_ckpt"], exp_configs["save_ckpt"], exp_configs["save_logits"]
    logger = Logger(log_configs["comet"], log_configs["pandas_df"], log_configs["wandb"], log_ckpt, save_ckpt, save_logits)
    logger.log(config, best_metrics, logits,
                model_name="bittransgnn_os_direct_induction", project_name="baseline-os-bittransgnn-direct-induction",
                ckpt_dir=ckpt_dir)
    count+=1
    print(f"Completed {count} out of {len(master_config['configs'])} configurations.")
