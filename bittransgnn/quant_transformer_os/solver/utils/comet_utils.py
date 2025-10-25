# pip install comet-ml
from comet_ml import Experiment
import os, json

def _resolve_config_to_dict(cfg):
    # works for OmegaConf or plain objects
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        # fallback: best-effort attribute/dict flatten
        def to_dict(x):
            if isinstance(x, dict): return {k: to_dict(v) for k, v in x.items()}
            if hasattr(x, "__dict__"): return {k: to_dict(v) for k, v in vars(x).items()}
            return x
        return to_dict(cfg)

def start_comet(project="ptq-bert", workspace=None, cfg=None, cfg_path=None, tags=None):
    exp = Experiment(
        project_name=project,
        workspace=workspace,                    # or None
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_output_logging="simple",
    )
    if tags: exp.add_tags(tags)
    # Log flattened config
    if cfg is not None:
        cfg_dict = _resolve_config_to_dict(cfg)
        exp.log_parameters(_flatten_dict(cfg_dict))
    # Upload the exact config file as an artifact
    if cfg_path and os.path.exists(cfg_path):
        exp.log_asset(cfg_path, file_name=os.path.basename(cfg_path))
    return exp

def _flatten_dict(d, parent_key="", sep="."):
    out = {}
    for k, v in (d or {}).items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out

def log_eval_to_comet(exp, split_name, metrics: dict, step=None):
    # e.g., metrics = {"accuracy": 0.85, "mcc": 0.40, "f1": 0.88}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            exp.log_metric(f"{split_name}/{k}", v, step=step)
