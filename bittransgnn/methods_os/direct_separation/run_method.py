import os
import torch
from transformers import AutoConfig

from pathlib import Path

from utils import get_model_type, get_train_state
from data.loader.dataloaders import TextDataObject
from models import BitTransformer, BitTransformerOS, BertForSeqClsNoPooler
from inference_engines import BitTransformerInference
#from utils_os import _swap_ln_to_lnplus, replace_qlinears_with_quantizedlayer, dict_to_namespace, DictNamespace # utils_os
from quant_transformer_os.solver.quant_model import quantize_model
from quant_transformer_os.solver.utils.glue_utils import parse_config
from quant_transformer_os.quantization.state import enable_calibration_woquantization, enable_quantization,\
        disable_all, enable_calibration_quantization, enable_calibration_quantization_plus, set_observer_name  # noqa: F401
from utils_os import bypass_fakequant, swap_ln_to_lnplus, replace_qlinears_with_quantizedlayer, dict_to_namespace, DictNamespace # utils_os

def run_bittransgnn_for_direct_separation(config):
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
        quant_bit = model_configs.get("quant_bit", None)

        repo_root = Path(__file__).resolve().parents[1]  # /.../BitTransGNN/bittransgnn/methods_os/
        if baseline_type == "os":
                os_config_path = repo_root / f"os_configs/bert_ptq/{int(quant_bit)}-bit/twc_fine_gamma/{dataset_name}/config.yaml"
        else:
                os_config_path = repo_root / f"os_configs/bert_ptq/{int(quant_bit)}-bit/{baseline_type}/{dataset_name}/config.yaml"

        os_config = parse_config(str(os_config_path))  # Convert Path to string if needed

        # Convert checkpoint_dir to Path object if it's a string
        if isinstance(checkpoint_dir, str):
                checkpoint_dir = Path(checkpoint_dir)
                print("checkpoint_dir: ", checkpoint_dir)
        
        # Convert manual_load_ckpt to Path object if it's a string
        if isinstance(manual_load_ckpt, str):
                manual_load_ckpt = Path(manual_load_ckpt)
                print("manual_load_ckpt: ", manual_load_ckpt)

        #quant_name = get_quant_name(num_states)
        train_state = get_train_state(joint_training)
        model_type = get_model_type(quantize_bert=True, quantize_gcn=quantize_gcn)

        text_data = TextDataObject(dataset_name, batch_size, adj_type=adj_type, seed=seed)
        nb_class = text_data.nb_class

        data_path = checkpoint_dir.joinpath(f"./direct_induction/{baseline_type}_{dataset_name}_{int(quant_bit)}_bit")
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

        if joint_training:
                load_ckpt_dir = manual_load_ckpt.joinpath(f"./bittransgnn/{baseline_type}_{dataset_name}_{int(quant_bit)}_bit/model_ckpts/checkpoint.pth")
        else:
                load_ckpt_dir = manual_load_ckpt.joinpath(f"./bittransgnn/{baseline_type}_{dataset_name}_{int(quant_bit)}_bit_static/model_ckpts/checkpoint.pth")
        #ckpt = get_pretrained_bert_ckpt(model_type, bert_pre_model,
        ckpt = torch.load(load_ckpt_dir, map_location=torch.device('cpu'))

        regression = dataset_name == "stsb"

        config_org = AutoConfig.from_pretrained("bert-base-uncased", num_labels=nb_class)

        org_model = BertForSeqClsNoPooler(config_org)

        qmodel = quantize_model(org_model, os_config)

        #print("ckpt")
        #print(ckpt)

        #fixed_state_dict = remap_for_target_model(ckpt["state_dict"], target_model=QuantizedBertForSeqClsNoPooler(org_model, w_qconfig_obj, a_qconfig_obj))
        #fixed_state_dict = ckpt["state_dict"]
        sd_bert = ckpt["bert_model"]
        sd_cls = ckpt["classifier"]
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
        enable_quantization(qmodel)   # your repo’s helper

        #loaded_model = QuantizedBertForSeqClsNoPooler(org_model, w_qconfig_obj, a_qconfig_obj)

        # Load with strict=False to handle any remaining mismatches
        miss_b, unexp_b = qmodel.bert.load_state_dict(sd_bert, strict=False)
        miss_c, unexp_c = qmodel.classifier.load_state_dict(sd_cls, strict=False)
        print(f"[BERT] missing={len(miss_b)} unexpected={len(unexp_b)}")
        print(f"[CLS ] missing={len(miss_c)} unexpected={len(unexp_c)}")

        print("miss_b")
        print(miss_b)
        print("unexp_b")
        print(unexp_b)
        print("miss_c")
        print(miss_c)
        print("unexp_c")
        print(unexp_c)

        bert_model = qmodel.bert
        #print("bert_model")
        #print(bert_model)
        classifier = qmodel.classifier
        #print("classifier")
        #print(classifier)

        model = BitTransformerOS(bert_model=bert_model, classifier=classifier, regression=regression)
        model = model.to(device)
        text_data.set_dataloaders_bert(model, max_length)

        inference_engine = BitTransformerInference(model, dataset_name, text_data, device)
        inference_metrics, logits = inference_engine.run(report_time)
        return inference_metrics, ckpt_dir_dict, logits


"""
# alternative way if there is an issue

    # 1) rebuild the same OS+ quant graph you used during training
    nb_class = text_data.nb_class
    cfg = ckpt.get("config", os_config)  # prefer embedded cfg if present

    base_cfg = AutoConfig.from_pretrained("bert-base-uncased", num_labels=nb_class)
    base = BertForSeqClsNoPooler(base_cfg)

    qmodel = quantize_model(base, cfg)   # reconstruct quant wrappers

    # Apply the same OS+ edits you used at training time:
    if ckpt["arch"].get("osplus_lnplus", True):
        _swap_ln_to_lnplus(qmodel)
    if ckpt["arch"].get("wrapped_linears", True):
        replace_qlinears_with_quantizedlayer(
            root=qmodel, w_qconfig=cfg.quant.w_qconfig, a_qconfig=cfg.quant.a_qconfig,
            qinput=True, activation=None, verbose=False
        )

    # 2) load tensors
    missing, unexpected = qmodel.bert.load_state_dict(ckpt["bert_state"], strict=False)
    print("bert missing:", missing, "unexpected:", unexpected)
    missing, unexpected = qmodel.classifier.load_state_dict(ckpt["classifier_state"], strict=False)
    print("clf  missing:", missing, "unexpected:", unexpected)

    qmodel.eval()
    from quant_transformer.quantization.state import enable_quantization
    enable_quantization(qmodel)

    model = BertClassifierOS(bert_model=qmodel.bert, classifier=qmodel.classifier, regression=regression)
    model.eval()
    enable_quantization(model)
"""