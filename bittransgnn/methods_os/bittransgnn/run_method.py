import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoConfig

from utils import get_quant_name, get_train_state, get_model_type, set_bittransgnn_ckpt_dir, get_pretrained_transformer_ckpt, load_bittransgnn_for_inference
from data.loader.dataloaders import GraphDataObject, TextDataObject
from models import BitTransGNNOS, BitTransGNN, BertForSeqClsNoPooler
from trainers import BitTransGNNTrainer
from inference_engines import BitTransGNNInference
from quantization.binarize_model import quantize_bittransgnn_architecture, quantize_bittransgnn_for_inference

from quant_transformer_os.model.quant_bert import QuantizedBertForSeqClsNoPooler
from quant_transformer_os.solver.quant_model import quantize_model
from quant_transformer_os.solver.utils.glue_utils import parse_config
from quant_transformer_os.quantization.state import enable_calibration_woquantization, enable_quantization,\
        disable_all, enable_calibration_quantization, enable_calibration_quantization_plus, set_observer_name  # noqa: F401
from utils_os import swap_ln_to_lnplus, replace_qlinears_with_quantizedlayer, dict_to_namespace, DictNamespace # utils_os
#from utils_os import bypass_fakequant

def _toggle_fakequant(module, enable_fake, enable_obs):
    # Works with PyTorch FakeQuantize + many custom fake-quant layers
    if hasattr(module, "fake_quant_enabled"):
        try:
            module.enable_fake_quant() if enable_fake else module.disable_fake_quant()
        except Exception:
            module.fake_quant_enabled[0] = torch.tensor(int(enable_fake), dtype=torch.uint8)

    if hasattr(module, "observer_enabled"):
        try:
            module.enable_observer() if enable_obs else module.disable_observer()
        except Exception:
            module.observer_enabled[0] = torch.tensor(int(enable_obs), dtype=torch.uint8)

    # Common custom names
    for attr in ["quant_enabled", "enable_quant", "disable_quant"]:
        if hasattr(module, attr):
            try:
                getattr(module, "enable_quant" if enable_fake else "disable_quant")()
            except Exception:
                try:
                    module.quant_enabled = enable_fake
                except Exception:
                    pass

def bypass_fakequant(model, bypass=True):
    for m in model.modules():
        _toggle_fakequant(m, enable_fake=not bypass, enable_obs=not bypass)

def run_bittransgnn(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_gcn, gcn_num_states = model_configs["max_length"], model_configs["quantize_gcn"], model_configs["gcn_num_states"]
    dataset_name, bert_pre_model, device, report_time, checkpoint_dir, nb_epochs, patience = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"], exp_configs["checkpoint_dir"], exp_configs["nb_epochs"], exp_configs["patience"]
    bert_lr, gcn_lr, batch_size, joint_training, dropout, n_hidden = parameters["bert_lr"], parameters["gcn_lr"], parameters["batch_size"], parameters["joint_training"], parameters["dropout"], parameters["graph_hidden_size"]
    lmbd, gcn_layers = parameters["lmbd"], parameters["gcn_layers"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    save_ckpt = exp_configs["save_ckpt"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    inference_type = exp_configs["inference_type"]
    eval_test, eval_test_every_n_epochs = exp_configs["eval_test"], exp_configs["eval_test_every_n_epochs"]
    seed = exp_configs["seed"]
    adj_type=config["model_configs"].get("adj_type", None)
    baseline_type = exp_configs.get("baseline_type", None)
    quant_bit = model_configs.get("quant_bit", 8)

    print("baseline_type: ", baseline_type)

    #config_dir = Path(__file__).parent / "configs"
    repo_root = Path(__file__).resolve().parents[1]  # /.../BitTransGNN/bittransgnn/methods_os/
    if baseline_type == "os":
        os_config_path = repo_root / f"os_configs/bert_ptq/{int(quant_bit)}-bit/twc_fine_gamma/{dataset_name}/config.yaml"
    else:
        os_config_path = repo_root / f"os_configs/bert_ptq/{int(quant_bit)}-bit/{baseline_type}/{dataset_name}/config.yaml"

    os_config = parse_config(str(os_config_path))  # Convert Path to string if needed
    print("os_config_path: ", os_config_path)
    print("os_config: ", os_config)

    #os_config_dict = config["os_config"]
    #os_config = DictNamespace(os_config_dict)

    if inference_type == "inductive":
        inductive = True
        train_only = True
    else:
        inductive = False
        train_only = False

    #quant_name = get_quant_name(num_states)
    train_state = get_train_state(joint_training)
    model_type = get_model_type(quantize_bert=True, quantize_gcn=quantize_gcn)

    # Convert checkpoint_dir to Path object if it's a string
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
        print("checkpoint_dir: ", checkpoint_dir)
    
    # Convert manual_load_ckpt to Path object if it's a string
    if isinstance(manual_load_ckpt, str):
        manual_load_ckpt = Path(manual_load_ckpt)
        print("manual_load_ckpt: ", manual_load_ckpt)

    if joint_training:
        data_path = checkpoint_dir.joinpath(f"./bittransgnn/{baseline_type}_{dataset_name}_{int(quant_bit)}_bit")
    else:
        data_path = checkpoint_dir.joinpath(f"./bittransgnn/{baseline_type}_{dataset_name}_{int(quant_bit)}_bit_static")
    result_data_path = data_path.joinpath("./results")
    model_data_path = data_path.joinpath("./model_ckpts")
    logit_data_path = data_path.joinpath("./logits")
    ckpt_dir_dict = {"model_ckpt_dir": model_data_path, "result_ckpt_dir": result_data_path, "logit_ckpt_dir": logit_data_path}

    if save_ckpt:
        print("model_ckpt_dir", ckpt_dir_dict["model_ckpt_dir"])
        print("model_data_path", model_data_path)
        os.makedirs(ckpt_dir_dict["model_ckpt_dir"], exist_ok=True)
        os.makedirs(ckpt_dir_dict["result_ckpt_dir"], exist_ok=True)
        os.makedirs(ckpt_dir_dict["logit_ckpt_dir"], exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_load_dir = manual_load_ckpt.joinpath(f"./{baseline_type}_{dataset_name}_w{int(quant_bit)}_a{int(quant_bit)}/checkpoint.pth")
    #ckpt = get_pretrained_bert_ckpt(model_type, bert_pre_model,
    ckpt = torch.load(ckpt_load_dir, map_location=torch.device('cpu'))

    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset_name)
    #idx_loaders, nb_node, nb_train, nb_val, nb_test, nb_word, nb_class = get_dataloaders_bertgcn(dataset_name, batch_size, features, y_train, train_mask, val_mask, test_mask)
    
    graph_data = GraphDataObject(dataset_name, batch_size, adj_type=adj_type, seed=seed, train_only=train_only)
    
    nb_class = graph_data.nb_class
    regression = dataset_name == "stsb"

    if baseline_type == "os":
        config_org = AutoConfig.from_pretrained("bert-base-uncased", num_labels=nb_class, layer_norm_eps=1e-5)
    else:
        config_org = AutoConfig.from_pretrained("bert-base-uncased", num_labels=nb_class)

    org_model = BertForSeqClsNoPooler(config_org)

    qmodel = quantize_model(org_model, os_config)

    #fixed_state_dict = remap_for_target_model(ckpt["state_dict"], target_model=QuantizedBertForSeqClsNoPooler(org_model, w_qconfig_obj, a_qconfig_obj))
    fixed_state_dict = ckpt["state_dict"]
    #print('ckpt["state_dict"].keys()')
    #print(ckpt["state_dict"].keys())

    if baseline_type == "osplus":
        swap_ln_to_lnplus(qmodel)
        n_wrapped = replace_qlinears_with_quantizedlayer(
            root=qmodel,
            w_qconfig=os_config.quant.w_qconfig,
            a_qconfig=os_config.quant.a_qconfig,
            qinput=True,                 # add input activation fake-quant
            activation=None,             # keep BERT activations outside the linear
            verbose=True,
        )
    elif baseline_type == "os":
        if os_config.quant.ln.delay:
            from quant_transformer_os.solver.gamma_migration import delay_ln
            bypass_fakequant(qmodel, bypass=True)  # Disable all fake-quant and observers
            qmodel_cpu = qmodel.cpu().to(dtype=torch.float64)
            #qmodel = delay_ln(qmodel, os_config.quant, os_config.model)
            qmodel_cpu = delay_ln(qmodel_cpu, os_config.quant, os_config.model)
            qmodel = qmodel_cpu.to(dtype=torch.float32).to(device)
            bypass_fakequant(qmodel, bypass=False)  # Disable all fake-quant and observers
            #enable_calibration_woquantization(qmodel, quantizer_type='weight_fake_quant')
            #calibrate(trainer, fp_input)
            #enable_calibration_woquantization(qmodel, quantizer_type='act_fake_quant')
            #calibrate(trainer, fp_input)

    qmodel.eval()
    enable_quantization(qmodel)

    #loaded_model = QuantizedBertForSeqClsNoPooler(org_model, w_qconfig_obj, a_qconfig_obj)

    # Load with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = qmodel.load_state_dict(fixed_state_dict, strict=False)

    print("missing_keys")
    print(missing_keys)

    print("unexpected_keys")
    print(unexpected_keys)

    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys in checkpoint")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")

    bert_model = qmodel.bert
    #print("bert_model")
    #print(bert_model)
    classifier = qmodel.classifier
    #print("classifier")
    #print(classifier)
    nan_hits = []

    def check_nan(name):
        def _hook(_m, _in, out):
            with torch.no_grad():
                x = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (list, tuple)) else None)
                if x is not None and (torch.isnan(x).any() or torch.isinf(x).any()):
                    nan_hits.append(name)
                    # Optional: raise to stop early
                    # raise RuntimeError(f"NaN at {name}")
        return _hook

    # Add hooks to sensitive spots
    for name, module in qmodel.named_modules():
        if any(tag in name.lower() for tag in ["layernorm", "ln", "attn", "attention", "softmax", "gelu", "classifier"]):
            module.register_forward_hook(check_nan(name))

    """
    qmodel.eval().cuda()
    with torch.no_grad():
        # run a tiny batch through (1-2 samples)
        _ = qmodel(**your_minibatch_on_cuda)
    
    """

    model = BitTransGNNOS(bert_model=bert_model, classifier=classifier, pretrained_model=bert_pre_model, joint_training=joint_training,
                         quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                         nb_class=nb_class, lmbd=lmbd, gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                         regression=regression)#,w_qconfig=w_qconfig, a_qconfig=a_qconfig, qoutput=True)
    graph_data.set_transformer_data(model, max_length)
    graph_data.set_graph_data(model)

    model = model.to(device)

    lr = gcn_lr
    if joint_training:
        optimizer = torch.optim.Adam([
                {'params': model.bert_model.parameters(), 'lr': bert_lr},
                {'params': model.classifier.parameters(), 'lr': bert_lr},
                {'params': model.gcn.parameters(), 'lr': gcn_lr},
            ], lr=lr
        )
    else:
        optimizer = torch.optim.Adam([
                #{'params': model.classifier.parameters(), 'lr': bert_lr},
                {'params': model.gcn.parameters(), 'lr': gcn_lr},
            ], lr=lr
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[nb_epochs//2], gamma=0.1)

    trainer = BitTransGNNTrainer(model, dataset_name, optimizer, scheduler, graph_data, joint_training, device, batch_size, inductive, eval_test, eval_test_every_n_epochs)
    model_checkpoint, best_metrics, logits = trainer.run(nb_epochs, patience, report_time)
    return model_checkpoint, ckpt_dir_dict, best_metrics, logits

