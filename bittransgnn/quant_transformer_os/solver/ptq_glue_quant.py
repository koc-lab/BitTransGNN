import os
import numpy as np
import logging
import sys
import argparse
import transformers
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
)
import evaluate as evaluate_hf
import datasets
import random
from datasets import load_metric
import torch  # noqa E401
import torch.fx
import quant_transformer_os.solver.utils.glue_utils as glue_utils
from quant_transformer_os.quantization.state import enable_calibration_woquantization, enable_quantization,\
        disable_all, enable_calibration_quantization, enable_calibration_quantization_plus, set_observer_name  # noqa: F401
from quant_transformer_os.quantization.observer import ObserverBase  # noqa: F401
from quant_transformer_os.quantization.fake_quant import LSQPlusFakeQuantize, QuantizeBase  # noqa: F401
from quant_model import quantize_model
import token_wise_clipping
logger = logging.getLogger("transformer")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)  # 0 is now the only visible device (which is physical GPU 1)

print("Current CUDA device:", torch.cuda.current_device())
print("CUDA device count:", torch.cuda.device_count())

import torch.nn as nn
import torch.nn.functional as F  # ensure this is imported since LNPlus.forward uses F.layer_norm
from quant_transformer_os.model.util_layernorm import QuantizedLayerNorm, QuantizedLayerNormPlus

def _swap_ln_to_lnplus(module: nn.Module):
    """
    Recursively:
      - Replace any nn.LayerNorm with QuantizedLayerNormPlus(child)
      - If a QuantizedLayerNorm is found, upgrade its inner .layernorm to LNPlus
    """
    for name, child in list(module.named_children()):
        # Case A: your quant wrapper
        if isinstance(child, QuantizedLayerNorm):
            ln = child.layernorm
            if isinstance(ln, nn.LayerNorm) and not isinstance(ln, QuantizedLayerNormPlus):
                # replace the inner layernorm in-place
                lnplus = QuantizedLayerNormPlus(ln)   # <-- pass the module, not eps/shape
                child.layernorm = lnplus
            # keep recursing in case there are nested modules
            _swap_ln_to_lnplus(child)

        # Case B: plain nn.LayerNorm
        elif isinstance(child, nn.LayerNorm) and not isinstance(child, QuantizedLayerNormPlus):
            lnplus = QuantizedLayerNormPlus(child)   # <-- correct call
            # safely replace this child on the parent
            module._modules[name] = lnplus

        else:
            _swap_ln_to_lnplus(child)

# --- put these near your other imports ---
import torch
import torch.nn as nn
from quant_transformer_os.quantization.quantized_module import QuantizedModule, QuantizedLayer, Quantizer, QLinear

def _looks_like_quant_linear(m: nn.Module) -> bool:
    """
    Heuristic: a quantized 'linear-like' module if:
      - it's a QuantizedModule
      - has a 2D 'weight' Parameter (out_features x in_features)
      - is NOT an embedding (padding_idx attribute is typical for embeddings)
      - does NOT already have an input activation fake-quant (act_fake_quant)
    """
    if not isinstance(m, QuantizedModule):
        return False

    # exclude embeddings or other non-linear weight modules
    if getattr(m, "padding_idx", None) is not None:
        return False

    w = getattr(m, "weight", None)
    if not isinstance(w, torch.nn.Parameter):
        return False
    if w.ndim != 2:
        return False

    # if it already has input activation quant, no need to wrap again
    if hasattr(m, "act_fake_quant"):
        return False

    return True


def replace_qlinears_with_quantizedlayer(root: nn.Module,
                                         w_qconfig,
                                         a_qconfig,
                                         qinput: bool = True,
                                         activation: nn.Module = None,
                                         verbose: bool = True) -> int:
    """
    Recursively replaces every 'quantized linear-like' module under `root`
    with a QuantizedLayer(module=<old>, activation=..., w_qconfig=..., a_qconfig=..., qinput=...).

    Returns number of replacements.
    """
    replaced = []

    def _recurse(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name

            #print("parent", parent)  # --- IGNORE ---

            if isinstance(parent, QLinear) or _looks_like_quant_linear(child):
                # Wrap the existing quantized-linear module inside QuantizedLayer.
                wrapper = QuantizedLayer(module=child,
                                         activation=activation,
                                         w_qconfig=w_qconfig,
                                         a_qconfig=a_qconfig,
                                         qinput=qinput)
                setattr(parent, name, wrapper)
                replaced.append(full)
            else:
                _recurse(child, full)

    _recurse(root, "")

    if verbose:
        print(f"[OS+] Wrapped {len(replaced)} quantized linear modules with QuantizedLayer (qinput={qinput})")
        for r in replaced[:8]:
            print("  •", r)
        if len(replaced) > 8:
            print(f"  ... (+{len(replaced)-8} more)")
    return len(replaced)

def set_logger(config_progress):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config_progress.log_level
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

import re
from types import SimpleNamespace
from quant_transformer_os.model.util_layernorm import QuantizedLayerNormPlus

from types import SimpleNamespace
from quant_transformer_os.model.util_layernorm import QuantizedLayerNormPlus

def _qcfg_ns(qc):
    return SimpleNamespace(
        bit=qc.bit, symmetric=qc.symmetric, ch_axis=qc.ch_axis,
        quantizer=getattr(qc, 'quantizer', None),
        group_size=getattr(qc, 'group_size', None),
    )

def wire_osplus_for_bert(model, config):
    name_to_mod = dict(model.named_modules())
    a_qcfg = _qcfg_ns(config.quant.a_qconfig)
    w_qcfg = _qcfg_ns(config.quant.w_qconfig)

    # attach maps/qcfgs and clear targets
    for name, m in name_to_mod.items():
        if isinstance(m, QuantizedLayerNormPlus):
            m._name_to_module_map = name_to_mod
            m.osplus_a_qcfg = a_qcfg
            m.osplus_w_qcfg = w_qcfg
            m._qualified_name = name
            m.osplus_target_linear_names = []

    wired = 0
    for name, m in name_to_mod.items():
        if not isinstance(m, QuantizedLayerNormPlus):
            continue

        # strip inner '.layernorm' if present
        base = name[:-10] if name.endswith(".layernorm") else name

        # ONLY wire attention.output.LayerNorm -> intermediate.dense
        if ".attention.output.LayerNorm" in base:
            fc1 = base.replace("attention.output.LayerNorm", "intermediate.dense")
            if fc1 in name_to_mod:
                m.osplus_target_linear_names.append(fc1)
                wired += 1

    print(f"wire_osplus_for_bert: wired {wired} attention.output.LayerNorm → intermediate.dense")


import json
from pathlib import Path

def save_full_model(model, save_path, cfg=None, save_config=True):
    """
    Saves a single-file checkpoint with:
      - model.state_dict()  (works for both FP and quantized models)
      - minimal metadata
      - optional serialized config alongside
    """
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    save_path_final = os.path.join(save_path, "checkpoint.pth")

    # Save the full state dict
    torch.save(
        {
            "state_dict": model.state_dict(),   # includes quantizer buffers if quantized
            "meta": {
                "arch": model.__class__.__name__,
                "backbone_prefix": getattr(model, "base_model_prefix", "bert"),
            },
        },
        save_path_final,
    )

    # (optional) save a JSON of your resolved config next to it
    if save_config and cfg is not None:
        cfg_json = f"{save_path}/config.json"
        def to_dict(x):
            if isinstance(x, dict): return {k: to_dict(v) for k, v in x.items()}
            if hasattr(x, "__dict__"): return {k: to_dict(v) for k, v in vars(x).items()}
            return x
        with open(cfg_json, "w", encoding="utf-8") as f:
            json.dump(to_dict(cfg), f, indent=2)
    return save_path_final


def make_huggingface_training_args(config_train, config_progress):
    training_args = TrainingArguments(
        seed=config_train.seed,
        output_dir=config_train.output_dir,
        overwrite_output_dir=config_train.overwrite_output_dir,
        do_train=config_train.do_train,
        do_eval=config_train.do_eval,
        do_predict=config_train.do_predict,
        evaluation_strategy=config_train.evaluation_strategy,
        eval_steps=config_train.eval_steps,
        per_device_train_batch_size=config_train.per_device_train_batch_size,
        per_device_eval_batch_size=config_train.per_device_eval_batch_size,
        gradient_accumulation_steps=config_train.gradient_accumulation_steps,
        eval_accumulation_steps=config_train.eval_accumulation_steps,
        learning_rate=config_train.learning_rate,
        weight_decay=config_train.weight_decay,
        max_grad_norm=config_train.max_grad_norm,
        num_train_epochs=config_train.num_train_epochs,
        max_steps=config_train.max_steps,
        lr_scheduler_type=config_train.lr_scheduler_type,
        warmup_ratio=config_train.warmup_ratio,
        warmup_steps=config_train.warmup_steps,
        gradient_checkpointing=config_train.gradient_checkpointing,
        log_level=config_progress.log_level,
        log_level_replica=config_progress.log_level_replica,
        logging_dir=config_progress.logging_dir,
        logging_strategy=config_progress.logging_strategy,
        logging_steps=config_progress.logging_steps,
        save_strategy=config_progress.save_strategy,
        save_steps=config_progress.save_steps,
        save_total_limit=config_progress.save_total_limit,
        save_on_each_node=config_progress.save_on_each_node,
        no_cuda=config_progress.no_cuda,
        run_name=config_progress.run_name,
        disable_tqdm=config_progress.disable_tqdm,
        load_best_model_at_end=config_progress.load_best_model_at_end,
        metric_for_best_model=config_progress.metric_for_best_model,
        greater_is_better=config_progress.greater_is_better
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    config_progress.log_level = training_args.get_process_log_level()
    return training_args


def prepare_input_output(trainer, cali_data):
    logger.info('**prepare fp input and output**')
    data_loader = trainer.get_eval_dataloader(cali_data)
    fp_input, fp_output = [], []
    with torch.no_grad():
        for p in data_loader:
            tmp = {}
            for k, v in p.items():
                tmp[k] = v.cuda()
            del tmp['labels']
            output = trainer.model(**tmp)[0].detach()
            fp_input.append(tmp)
            fp_output.append(output)
    return fp_input, fp_output


def calibrate(trainer, fp_input):
    logger.info("*** Calibrate ***")
    with torch.no_grad():
        for batch in fp_input:
            trainer.model(**batch)


def evaluate(trainer, eval_datasets):
    logger.info("*** Evaluate ***")
    if not isinstance(eval_datasets, tuple):
        eval_datasets = [eval_datasets]
    for i, sample in enumerate(eval_datasets[0]):
        if not (sample.get("sentence", None) and (sample.get("sentence1", None) and sample.get("sentence2", None))) or (isinstance(sample.get("sentence", None), float) and np.isnan(sample.get("sentence", None))):
            print(f"Invalid sample at index {i}: {sample}")
    metrics = []
    for i in range(len(eval_datasets)):
        metric = trainer.evaluate(eval_dataset=eval_datasets[i])
        metrics.append(metric)
    for i in range(len(metrics)):
        trainer.log_metrics("eval", metrics[i])
        trainer.save_metrics("eval", metrics[i])
    return metrics[0]  # return the first one for val set


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def simple_accuracy(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return {"accuracy": (preds == labels).mean().item()}

class _AccuracyMetric:
    """Mimics evaluate.Metric for accuracy: use metric.compute(predictions=..., references=...)."""
    def compute(self, predictions, references):
        preds = np.asarray(predictions)
        refs  = np.asarray(references)
        # if logits provided, convert to class ids
        if preds.ndim > 1:
            preds = preds.argmax(axis=-1)
        return {"accuracy": float((preds == refs).mean())}

def calibrate_batch(model, fp_input):
    logger.info("*** Calibrate ***")
    with torch.no_grad():
        for batch in fp_input:
            model(**batch)

def main(config_path):
    config = glue_utils.parse_config(config_path)
    set_seed(config.train.seed)
    if config.data.task_name == 'cola':
        config.progress.metric_for_best_model = 'matthews_correlation'
    elif config.data.task_name == 'stsb':
        config.progress.metric_for_best_model = 'pearson'
    else:
        config.progress.metric_for_best_model = 'accuracy'
    training_args = make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets, num_labels, label_list = glue_utils.load_dataset_labels(config.data)
    #tokenizer, model = glue_utils.load_model(config.model, config.data, num_labels)
    tokenizer, model = glue_utils.load_model(config.model, config.data, num_labels)
    """
    tokenizer, model = glue_utils.load_model(config.model, config.data, num_labels)
    if config.data.task_name not in ['20ng', 'mr', 'ohsumed', 'R8', 'R52']:
        raw_datasets, num_labels, label_list = glue_utils.load_dataset_labels(config.data)
    else:
        raw_datasets = glue_utils.load_dataset(config.data)    
    """

    # label2id & id2label
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and config.data.task_name is not None
        and config.data.task_name != 'stsb'
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    elif config.data.task_name is not None and config.data.task_name != 'stsb':
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    # max_seq_length
    config.data.max_seq_length = min(config.data.max_seq_length, tokenizer.model_max_length)

    # work with datasets, preprocess first then get train/val/test one
    raw_datasets = glue_utils.preprocess_dataset(config.data, training_args, raw_datasets, label_to_id, tokenizer)
    # train_dataset, val_dataset, predict_datasets
    train_datasets = glue_utils.check_return_data(raw_datasets, 'train', True, config.data.max_train_samples)

    if config.data.task_name == 'mnli':
        eval_datasets = (
            glue_utils.check_return_data(raw_datasets, 'validation_matched', True, config.data.max_eval_samples),
            glue_utils.check_return_data(raw_datasets, 'validation_mismatched', True, config.data.max_eval_samples),
        )
    else:
        eval_datasets = (glue_utils.check_return_data(raw_datasets, 'validation', True, config.data.max_eval_samples), )

    test_datasets = (glue_utils.check_return_data(raw_datasets, 'test', True, config.data.max_eval_samples), )

    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_MODULES_CACHE"] = os.path.join(os.environ["HF_HOME"], "modules")
    os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)

    print(os.getenv("HF_MODULES_CACHE"))

    from sklearn.metrics import accuracy_score
    if config.data.task_name in ["20ng", "mr", "ohsumed", "R8", "R52"]:
        #metric = evaluate_hf.load("evaluate-metric/accuracy")
        #metric = evaluate_hf.load("accuracy")
        #metric = _AccuracyMetric()
        metric = accuracy_score
    else:
        metric = load_metric("glue", config.data.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if config.data.is_regression else np.argmax(preds, axis=1)

        if metric != accuracy_score:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
        else:
            result = accuracy_score(p.label_ids, preds)
            result = {"accuracy": result}  # wrap in dict for consistency
        return result
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if config.data.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    model.eval()
    model.cuda()

    trainer = Trainer(
        model=model,
        #model=fp_model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets[0],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    cali_data = train_datasets.shuffle(seed=config.train.seed).select(range(config.quant.calibrate))
    fp_input, fp_output = prepare_input_output(trainer, cali_data)
    USE_OSPLUS = bool(getattr(getattr(config.quant, "osplus", None), "enable", False))
    print("USE_OSPLUS: ", USE_OSPLUS)

    if getattr(config, "quant", None):
        #print("model before quantize")
        #print(model)
        model = quantize_model(model, config)     # <- important: before OS+
        #print("model after quantize")
        #print(model)
        model.eval().cuda()
        trainer.model = model

        from quant_transformer_os.quantization.state import disable_all, enable_calibration_woquantization
        import quant_transformer_os.quantization.migration as migration
        from quant_transformer_os.quantization.quantized_module import QuantizedModule
        from quant_transformer_os.model.util_layernorm import QuantizedLayerNorm, QuantizedLayerNormPlus

        if USE_OSPLUS:
            # 1) Ensure all LayerNorm are LNPlus and wire LNPlus -> fc1 targets
            _swap_ln_to_lnplus(model)
            n_wrapped = replace_qlinears_with_quantizedlayer(
                root=trainer.model,
                w_qconfig=config.quant.w_qconfig,
                a_qconfig=config.quant.a_qconfig,
                qinput=True,                 # add input activation fake-quant
                activation=None,             # keep BERT activations outside the linear
                verbose=True,
            )                        
            #print(model)

            enable_calibration_quantization_plus(model, except_quantizer=getattr(config.quant, 'except_quantizer', None))
            if hasattr(config.quant, 'migrate') and config.quant.migrate:
                for name, module in model.named_modules():
                    if isinstance(module, QuantizedModule):
                        module.set_cac_migrate(True)
                calibrate_batch(model, [fp_input[0]])
                for name, module in model.named_modules():
                    if isinstance(module, QuantizedModule):
                        module.set_cac_migrate(False)
                import quant_transformer_os.quantization.migration as migration
                migration.fuse_migration(model)
            calibrate_batch(model, fp_input)
            enable_quantization(model, except_quantizer=getattr(config.quant, 'except_quantizer', None))

        else:
            #model = quantize_model(model, config)
            #model.eval()
            #model.cuda()
            #trainer.model = model
            if config.quant.ln.delay:
                from gamma_migration import delay_ln
                trainer.model = delay_ln(trainer.model, config.quant, config.model)
            # calibrate the weight
            enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
            #calibrate(trainer, [fp_input[0]])
            calibrate(trainer, fp_input)
            if 'PruneMinMaxObserver' in config.quant.a_qconfig.observer:
                disable_all(trainer.model)
                set_observer_name(trainer.model)
                token_wise_clipping.token_wise_clipping(trainer, fp_input, fp_output, config)
                if 'LSQ' in config.quant.a_qconfig.quantizer:
                    token_wise_clipping.learn_scale(trainer, fp_input, fp_output,
                                                    getattr(config.quant, 'learn', {'lr': 1e-5, 'epoch': 3}))
            else:
                # calibrate the activation
                enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
                calibrate(trainer, fp_input)
            torch.cuda.empty_cache()

    if training_args.do_eval:
        if getattr(config, 'quant', None):
            enable_quantization(trainer.model)
        torch.cuda.empty_cache()
        print("***** Val *****")
        valid_metrics = evaluate(trainer, eval_datasets)

    if training_args.do_predict:
        if getattr(config, 'quant', None):
            enable_quantization(trainer.model)
        torch.cuda.empty_cache()
        print("***** Test *****")
        test_metrics = evaluate(trainer, test_datasets)

    from quant_transformer_os.solver.utils.comet_utils import start_comet, log_eval_to_comet

    # after you build/save ckpts:
    comet = start_comet(
        project="os_ptq_baselines",
        workspace=None,  # add your workspace string
        cfg=config,
        #cfg_path=paths["config_path"] if isinstance(paths := save_ckpt_pair(...), dict) else None,
        tags=[f"task:{config.data.task_name}", f"bits:W{config.quant.w_qconfig.bit}A{config.quant.a_qconfig.bit}",
            "OSplus" if USE_OSPLUS else "OS"]
    )

    # Suppose evaluate() returns a dict of metrics:
    #valid_metrics = evaluate(trainer, eval_datasets)  # your existing call
    log_eval_to_comet(comet, "valid", valid_metrics)

    # For test set:
    #test_metrics = evaluate(trainer, test_datasets)
    log_eval_to_comet(comet, "test", test_metrics)

    comet.end()

    osplus_flag = "osplus" if USE_OSPLUS else "os"

    ckpt_save_dir = os.path.join(config.train.model_ckpt_save_dir, f"{osplus_flag}_{config.data.task_name}_w{config.quant.w_qconfig.bit}_a{config.quant.a_qconfig.bit}")
    print("ckpt_save_dir")
    print(ckpt_save_dir)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    final_save_path = save_full_model(model=model, save_path=ckpt_save_dir, cfg=config, save_config=True)
    print("final_save_path")
    print(final_save_path)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
