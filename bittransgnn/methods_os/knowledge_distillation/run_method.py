import os

import torch

from pathlib import Path

from utils import get_train_state, get_model_type
from data.loader.dataloaders import load_corpus, GraphDataObject, TextDataObject
from models import BitTransGNN, BitTransformerStudent, BitTransGNNOS, BitTransformerStudentOS, BertForSeqClsNoPooler
from trainers import BitTransGNNKDTrainer
from inference_engines import BitTransformerInference
from quantization.binarize_model import quantize_teacher_architecture, quantize_student_architecture
from quant_transformer_os.solver.utils.glue_utils import parse_config
from quant_transformer_os.quantization.quantized_module import QuantizedModule, QuantizedLayer, Quantizer, QLinear

from transformers import AutoConfig

from quant_transformer_os.model.quant_bert import QuantizedBertForSeqClsNoPooler
from quant_transformer_os.solver.quant_model import quantize_model
from quant_transformer_os.solver.utils.glue_utils import parse_config
from quant_transformer_os.quantization.state import enable_calibration_woquantization, enable_quantization,\
        disable_all, enable_calibration_quantization, enable_calibration_quantization_plus, set_observer_name  # noqa: F401
from utils_os import bypass_fakequant, swap_ln_to_lnplus, replace_qlinears_with_quantizedlayer, dict_to_namespace, DictNamespace # utils_os

def run_bittransgnn_kd(config):
    exp_configs = config["experiment_configs"]
    model_configs = config["model_configs"]
    parameters = config["parameters"]
    load_configs = config["load_configs"]
    max_length, quantize_gcn, gcn_num_states = model_configs["max_length"], model_configs["quantize_gcn"], model_configs["gcn_num_states"]
    dataset_name, bert_pre_model, device, report_time, checkpoint_dir, nb_epochs, patience = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"], exp_configs["checkpoint_dir"], exp_configs["nb_epochs"], exp_configs["patience"]
    bert_lr, gcn_lr, batch_size, joint_training, dropout, n_hidden = parameters["bert_lr"], parameters["gcn_lr"], parameters["batch_size"], parameters["joint_training"], parameters["dropout"], parameters["graph_hidden_size"]
    teacher_lmbd, gcn_layers = parameters["teacher_lmbd"], parameters["gcn_layers"]
    adj_type = model_configs["adj_type"]
    save_ckpt = exp_configs["save_ckpt"]
    local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
    distillation_type = exp_configs["distillation_type"]
    workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
    alpha_d, temperature = parameters["alpha_d"], parameters["temperature"]
    inference_type = exp_configs["inference_type"]
    eval_test, eval_test_every_n_epochs = exp_configs["eval_test"], exp_configs["eval_test_every_n_epochs"]
    seed = exp_configs["seed"]
    baseline_type = exp_configs.get("baseline_type", None)
    quant_bit = model_configs.get("quant_bit", None)

    #config_dir = Path(__file__).parent / "configs"
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

    if inference_type == "inductive":
        inductive = True
        train_only = True
    else:
        inductive = False
        train_only = False

    #quant_name = get_quant_name(num_states)
    train_state = get_train_state(joint_training)
    model_type = get_model_type(quantize_bert=True, quantize_gcn=quantize_gcn)

    data_path = checkpoint_dir.joinpath(f"./bittransgnn-kd/{baseline_type}_{dataset_name}_{int(quant_bit)}_bit")
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

    student_ckpt_load_dir = manual_load_ckpt.joinpath(f"./{baseline_type}_{dataset_name}_w{int(quant_bit)}_a{int(quant_bit)}/checkpoint.pth")
    student_ckpt = torch.load(student_ckpt_load_dir, map_location="cpu")
    print("Transformer type: ", bert_pre_model)

    teacher_ckpt_load_dir = checkpoint_dir.joinpath(f"./bittransgnn/{baseline_type}_{dataset_name}_{int(quant_bit)}_bit/model_ckpts/checkpoint.pth")
    teacher_ckpt = torch.load(teacher_ckpt_load_dir, map_location="cpu")

    if teacher_lmbd == -1 or teacher_lmbd is None:
        teacher_lmbd = teacher_ckpt["lmbd"]
    else:
        assert (teacher_lmbd == teacher_ckpt["lmbd"])

    print(f"teacher_lmbd: {teacher_lmbd}")

    graph_data = GraphDataObject(dataset_name, batch_size, adj_type=adj_type, seed=seed, train_only=train_only)
    nb_class = graph_data.nb_class

    config_org = AutoConfig.from_pretrained("bert-base-uncased", num_labels=nb_class)

    org_model = BertForSeqClsNoPooler(config_org)

    qmodel_student = quantize_model(org_model, os_config)
    qmodel_teacher = quantize_model(org_model, os_config)

    student_fixed_state_dict = student_ckpt["state_dict"]
    #print('ckpt["state_dict"].keys()')
    #print(ckpt["state_dict"].keys())

    if baseline_type == "osplus":
        swap_ln_to_lnplus(qmodel_student)
        swap_ln_to_lnplus(qmodel_teacher)
        n_wrapped = replace_qlinears_with_quantizedlayer(
            root=qmodel_student,
            w_qconfig=os_config.quant.w_qconfig,
            a_qconfig=os_config.quant.a_qconfig,
            qinput=True,                 # add input activation fake-quant
            activation=None,             # keep BERT activations outside the linear
            verbose=True,
        )
        n_wrapped = replace_qlinears_with_quantizedlayer(
            root=qmodel_teacher,
            w_qconfig=os_config.quant.w_qconfig,
            a_qconfig=os_config.quant.a_qconfig,
            qinput=True,                 # add input activation fake-quant
            activation=None,             # keep BERT activations outside the linear
            verbose=True,
        )
    elif baseline_type == "os":
        if os_config.quant.ln.delay:
            from quant_transformer_os.solver.gamma_migration import delay_ln
            bypass_fakequant(qmodel_student, bypass=True)  # Disable all fake-quant and
            qmodel_student_cpu = qmodel_student.cpu().to(dtype=torch.float64)
            qmodel_student_cpu = delay_ln(qmodel_student_cpu, os_config.quant, os_config.model)
            qmodel_student = qmodel_student_cpu.to(dtype=torch.float32).to(device)
            bypass_fakequant(qmodel_student, bypass=False)  # Disable all fake-quant and observers
            bypass_fakequant(qmodel_teacher, bypass=True)  # Disable all fake-quant and
            qmodel_teacher_cpu = qmodel_teacher.cpu().to(dtype=torch.float64)
            qmodel_teacher_cpu = delay_ln(qmodel_teacher_cpu, os_config.quant, os_config.model)
            qmodel_teacher = qmodel_teacher_cpu.to(dtype=torch.float32).to(device)
            bypass_fakequant(qmodel_teacher, bypass=False)  # Disable all fake-quant and
            #qmodel_student = delay_ln(qmodel_student, os_config.quant, os_config.model)
            #enable_calibration_woquantization(qmodel_student, quantizer_type='weight_fake_quant')
            #enable_calibration_woquantization(qmodel_student, quantizer_type='act_fake_quant')
            #qmodel_teacher = delay_ln(qmodel_teacher, os_config.quant, os_config.model)
            #enable_calibration_woquantization(qmodel_teacher, quantizer_type='weight_fake_quant')
            #enable_calibration_woquantization(qmodel_teacher, quantizer_type='act_fake_quant')

    qmodel_student.eval()
    enable_quantization(qmodel_student)   # your repo’s helper
    qmodel_teacher.eval()
    enable_quantization(qmodel_teacher)   # your repo’s helper

    missing_keys, unexpected_keys = qmodel_student.load_state_dict(student_fixed_state_dict, strict=False)

    print("missing_keys")
    print(missing_keys)

    print("unexpected_keys")
    print(unexpected_keys)

    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys in checkpoint")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")

    student_bert = qmodel_student.bert
    #print("bert_model")
    #print(bert_model)
    student_classifier = qmodel_student.classifier
    #print("classifier")
    #print(classifier)

    miss_b, unexp_b = qmodel_teacher.bert.load_state_dict(teacher_ckpt["bert_model"], strict=False)
    miss_c, unexp_c = qmodel_teacher.classifier.load_state_dict(teacher_ckpt["classifier"], strict=False)
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

    bert_model_teacher = qmodel_teacher.bert
    #print("bert_model")
    #print(bert_model)
    classifier_teacher = qmodel_teacher.classifier

    regression = dataset_name == "stsb"
    teacher_model = BitTransGNNOS(bert_model=bert_model_teacher, classifier=classifier_teacher, pretrained_model=bert_pre_model, joint_training=joint_training,
                         quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                         nb_class=nb_class, lmbd=teacher_lmbd, gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                         regression=regression)
    miss_gcn, unexp_gcn = teacher_model.gcn.load_state_dict(teacher_ckpt["gcn"])
    print(f"[GCN ] missing={len(miss_gcn)} unexpected={len(unexp_gcn)}")
    print("miss_gcn")
    print(miss_gcn)
    print("unexp_gcn")
    print(unexp_gcn)

    graph_data.set_transformer_data(teacher_model, max_length)
    graph_data.set_graph_data(teacher_model)

    student_model = BitTransformerStudentOS(bert_model=student_bert, classifier=student_classifier, nb_class=nb_class, regression=regression)

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    lr = gcn_lr
    teacher_optimizer = torch.optim.Adam([
            {'params': teacher_model.bert_model.parameters(), 'lr': bert_lr},
            {'params': teacher_model.classifier.parameters(), 'lr': bert_lr},
            {'params': teacher_model.gcn.parameters(), 'lr': gcn_lr},
        ], lr=lr
    )
    student_optimizer = torch.optim.Adam([
            {'params': student_model.bert_model.parameters(), 'lr': bert_lr},
            {'params': student_model.classifier.parameters(), 'lr': bert_lr},
        ], lr=bert_lr
    )
    teacher_scheduler = torch.optim.lr_scheduler.MultiStepLR(teacher_optimizer, milestones=[nb_epochs//2], gamma=0.1)
    student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, milestones=[nb_epochs//2], gamma=0.1)

    """
    if "cls_feats" in teacher_ckpt.keys():
        ext_cls_feats = teacher_ckpt["cls_feats"]
        print("The provided ext_cls_feats tensor from the model checkpoint will be used to initialize cls_feats.")
    else:
        ext_cls_feats = None
        print("No cls_feats tensor uploaded, the loaded BERT model instance will be used to initialize cls_feats.")
    """
    ext_cls_feats = None

    trainer = BitTransGNNKDTrainer(teacher_model, student_model, dataset_name,
                                  student_optimizer, student_scheduler, 
                                  graph_data,
                                  alpha_d, temperature, 
                                  device, 
                                  batch_size,
                                  distillation_type, 
                                  ext_cls_feats,
                                  teacher_optimizer, teacher_scheduler,
                                  inductive,
                                  eval_test, eval_test_every_n_epochs)
    model_checkpoint, best_metrics, logits = trainer.run(nb_epochs, patience, report_time)
    return model_checkpoint, ckpt_dir_dict, best_metrics, logits
