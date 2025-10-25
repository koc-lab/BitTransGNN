from pathlib import Path

import yaml

from .run_method import run_bittransformer
from logger import Logger
from utils import set_seed

from comet_ml import Optimizer

config_path = Path(__file__).parent.joinpath("./configs")
#config_file_path = config_path.joinpath("./sweep_config.yaml")
config_file_path = config_path.joinpath("./master_config.yaml")
with open(config_file_path) as file:
    master_config = yaml.load(file, Loader=yaml.FullLoader)
print("---list of configs in master_config---")
print(master_config["configs"])
for config in master_config["configs"]:
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    log_configs = config["log_configs"]
    parameters = config["parameters"]
    project_name, api_key, workspace = log_configs["project_name"], log_configs["api_key"], log_configs["workspace"]
    parameters_sweep = {par_name: parameters[par_name] for par_name in parameters.keys()}
    parameters_sweep["bert_pre_model"] = exp_configs["bert_pre_model"]
    parameters_sweep["num_states"] = model_configs["num_states"]
    parameters_sweep["quantize_bert"] = model_configs["quantize_bert"]
    parameters_sweep["quantize_embeddings"] = model_configs["quantize_embeddings"]
    parameters_sweep["quantize_attention"] = model_configs["quantize_attention"]
    parameters_sweep["num_bits_act"] = model_configs["num_bits_act"]
    parameters_sweep["dataset_name"] = exp_configs["dataset_name"]
    parameters_sweep["seed"] = exp_configs["seed"]
    parameters_sweep["adj_type"] = model_configs["adj_type"]  # Add adj_type to sweep parameters
    sweep_config = {"name": project_name, 
                    "algorithm": "grid",
                    "spec": {"metric": "best_val_accuracy", "objective": "maximize"}, 
                    "parameters": parameters_sweep}
    optimizer = Optimizer(sweep_config, 
                        api_key=api_key,
                        project_name=project_name,
                        workspace=workspace)
    for experiment in optimizer.get_experiments():
        set_seed(experiment.get_parameter("seed"))
        print(experiment)
        iter_parameters_sweep = {name: experiment.get_parameter(name) for name in parameters_sweep.keys()}
        iter_parameters = {name: experiment.get_parameter(name) for name in parameters.keys()}
        exp_configs["bert_pre_model"] = experiment.get_parameter("bert_pre_model")
        model_configs["num_states"] = experiment.get_parameter("num_states")
        model_configs["quantize_bert"] = experiment.get_parameter("quantize_bert")
        model_configs["quantize_embeddings"] = experiment.get_parameter("quantize_embeddings")
        model_configs["quantize_attention"] = experiment.get_parameter("quantize_attention")
        model_configs["num_bits_act"] = experiment.get_parameter("num_bits_act")
        exp_configs["seed"] = experiment.get_parameter("seed")
        exp_configs["dataset_name"] = experiment.get_parameter("dataset_name")
        model_configs["adj_type"] = experiment.get_parameter("adj_type")  # Get adj_type from experiment
        if experiment.get_parameter("num_states") == 0:
            quantize_bert = False
        else:
            quantize_bert = True
        model_configs["quantize_bert"] = quantize_bert
        iter_parameters["quantize_bert"] = quantize_bert
        iter_parameters_sweep["quantize_bert"] = quantize_bert
        exp_configs["seed"] = experiment.get_parameter("seed")
        print("iter_parameters_sweep")
        print(iter_parameters_sweep)
        print("iter_parameters")
        print(iter_parameters)
        iter_config = {"experiment_configs": exp_configs, "model_configs": model_configs, "parameters": iter_parameters}
        experiment.log_parameters(iter_config)
        model_checkpoint, ckpt_dir, best_metrics, logits = run_bittransformer(iter_config)
        log_ckpt, save_ckpt, save_logits = exp_configs["log_ckpt"], exp_configs["save_ckpt"], exp_configs["save_logits"]
        logger = Logger(log_configs["comet"], log_configs["pandas_df"], log_configs["wandb"], log_ckpt, save_ckpt, save_logits)
        logger.log(config, best_metrics, logits,
                    model_name="bittrans_train", project_name="bittrans_train",
                    model_checkpoint=model_checkpoint, ckpt_dir=ckpt_dir,
                    experiment=experiment, parameters=iter_parameters_sweep)
        experiment.end()
