from pathlib import Path

import yaml

from .run_method import run_bittransgnn_kd
from logger import Logger
from utils import get_model_type, get_train_state, set_seed

from comet_ml import Optimizer

config_path = Path(__file__).parent.joinpath("./configs")
config_file_path = config_path.joinpath("./sweep_config.yaml")
with open(config_file_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
exp_configs = config["experiment_configs"]
model_configs = config["model_configs"]
log_configs = config["log_configs"]
parameters = config["parameters"]
load_configs = config["load_configs"]
project_name, api_key, workspace = log_configs["project_name"], log_configs["api_key"], log_configs["workspace"]
parameters_sweep = {par_name: parameters[par_name] for par_name in parameters.keys()}
parameters_sweep["bert_pre_model"] = exp_configs["bert_pre_model"]
parameters_sweep["inference_type"] = exp_configs["inference_type"]
parameters_sweep["teacher_num_states"] = model_configs["teacher_num_states"]
parameters_sweep["student_num_states"] = model_configs["student_num_states"]
parameters_sweep["dataset_name"] = exp_configs["dataset_name"]
parameters_sweep["seed"] = exp_configs["seed"]
if parameters_sweep == "cola":
    sweep_metric = "best_val_matthews_corr"
elif parameters_sweep == "stsb":
    sweep_metric = "best_val_pearson_corr"
elif parameters_sweep == "mrpc":
    sweep_metric = "best_val_f1"
elif parameters_sweep == "rte":
    sweep_metric = "best_val_accuracy"
else:
    sweep_metric = "best_val_accuracy"
sweep_config = {"name": project_name, 
                "algorithm": "grid",
                "spec": {"metric": sweep_metric, "objective": "maximize"}, 
                "parameters": parameters_sweep}
optimizer = Optimizer(sweep_config, 
                      api_key=api_key,
                      project_name=project_name,
                      workspace=workspace)
for experiment in optimizer.get_experiments():
    set_seed(experiment.get_parameter("seed"))
    print(experiment)
    iter_parameters = {name: experiment.get_parameter(name) for name in parameters.keys()}
    iter_parameters_sweep = {name: experiment.get_parameter(name) for name in parameters_sweep.keys()}
    exp_configs["bert_pre_model"] = experiment.get_parameter("bert_pre_model")
    exp_configs["inference_type"] = experiment.get_parameter("inference_type")
    model_configs["teacher_num_states"] = experiment.get_parameter("teacher_num_states")
    model_configs["student_num_states"] = experiment.get_parameter("student_num_states")
    exp_configs["dataset_name"] = experiment.get_parameter("dataset_name")
    exp_configs["seed"] = experiment.get_parameter("seed")
    model_configs["model_type"] = get_model_type(model_configs["quantize_bert"], model_configs["quantize_gcn"])
    model_configs["train_state"] = get_train_state(parameters["joint_training"])
    print(iter_parameters_sweep)
    iter_config = {"experiment_configs": exp_configs, "model_configs": model_configs, "load_configs": load_configs, "parameters": iter_parameters}
    experiment.log_parameters(iter_config)
    model_checkpoint, ckpt_dir, best_metrics = run_bittransgnn_kd(iter_config)
    log_ckpt, save_ckpt = exp_configs["log_ckpt"], exp_configs["save_ckpt"]
    logger = Logger(log_configs["comet"], log_configs["pandas_df"], log_configs["wandb"], log_ckpt, save_ckpt)
    logger.log(config, best_metrics, 
                model_name="bittransgnn_kd", project_name="bittransgnn_kd", 
                model_checkpoint=model_checkpoint, ckpt_dir=ckpt_dir,
                experiment=experiment)
    experiment.end()
