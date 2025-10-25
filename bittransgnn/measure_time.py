
import os

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from comet_ml import Experiment

from data.loader.dataloaders import GraphDataObject, TextDataObject
from quantization.binarize_model import quantize_tf_model, replace_layer
from models import BitTransGNN, BitTransformer
from inference_engines import BitTransGNNInference, BitTransformerInference


# ------------ Config you can tweak ------------
WORKSPACE = None
PROJECT   = "bittransgnn-runtime"
API_KEY   = os.getenv("COMET_API_KEY", None)  # set this in your env
DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED      = 0

#DATASETS  = ["ohsumed", "cola", "20ng", "stsb"]          # <- iterate here
#DATASETS  = ["ohsumed", "cola", "20ng", "stsb", "mr", "rte", "mrpc"]          # <- iterate here
DATASETS = ["mr", "rte", "mrpc"]
ADJ_TYPES = ["full", "no_doc", "no_ww", "doc_doc"]      # <- and here

# Model / quant flags (shared across runs; feel free to move to loops)
bert_pre_model       = "bert-base-uncased"
batch_size           = 32
max_length           = 128
joint_training       = True
quantize_gcn         = False
gcn_num_states       = 2
num_states           = 2
num_bits_act         = 8.0
quantize_bert        = True
quantize_embeddings  = True
quantize_attention   = True
lmbd                 = 0.5
gcn_layers           = 2
n_hidden             = 256
dropout              = 0.5
train_only           = False

# Timing protocol
WARMUP_A  = 0  # Stage A (one-time): usually just run once
REPEAT_A  = 1
WARMUP_BC = 1  # Stage B/C: stabilize, then repeat a few times
REPEAT_BC = 5

# ------------ Repro ------------
def set_seed(seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------ Main loop ------------
for dataset_name in DATASETS:
    regression = (dataset_name == "stsb")

    for adj_type in ADJ_TYPES:
        set_seed(SEED)


        graph_data = GraphDataObject(dataset_name, batch_size, adj_type=adj_type, seed=SEED, train_only=train_only)

        text_data = TextDataObject(dataset_name, batch_size, adj_type=adj_type, seed=SEED)
        nb_class = text_data.nb_class

        bittransgnn_model = BitTransGNN(pretrained_model=bert_pre_model, joint_training=joint_training,
                                        quantize_gcn=quantize_gcn, gcn_num_states=gcn_num_states,
                                        nb_class=nb_class, lmbd=lmbd, gcn_layers=gcn_layers, n_hidden=n_hidden, dropout=dropout,
                                        regression=regression)
        graph_data.set_transformer_data(bittransgnn_model, max_length)
        graph_data.set_graph_data(bittransgnn_model)

        bittransgnn_model = quantize_tf_model(
            model=bittransgnn_model,
            num_states=num_states,
            linear_quantize=quantize_bert,
            quantize_embeddings=quantize_embeddings,
            num_bits_act=num_bits_act,
            quantize_attention=quantize_attention,
            attn_num_bits_act=num_bits_act)

        bittransgnn_model.classifier = replace_layer(
            module=bittransgnn_model.classifier,
            num_states=num_states,
            linear_quantize=quantize_bert,
            num_bits_act=num_bits_act)
        
        bittransgnn_model.to(DEVICE)

        bert_model = BitTransformer(pretrained_model=bert_pre_model, nb_class=nb_class, regression=regression)
        bert_model = quantize_tf_model(bert_model, num_states=num_states, linear_quantize=quantize_bert, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act, quantize_attention=quantize_attention, attn_num_bits_act=num_bits_act)
        bert_model = bert_model.to(DEVICE)
        text_data.set_dataloaders_bert(bert_model, max_length)

        # --- bittransgnn ---
        gcn_inf = BitTransGNNInference(
            model=bittransgnn_model,
            dataset_name=dataset_name,
            graph_data=graph_data,
            joint_training=False,
            device=DEVICE,
            ext_cls_feats=None,          # None to force timing BERT extraction
            batch_size=batch_size,
            inductive=False,
            recompute_bert=False
        )

        gcn_joint_inf = BitTransGNNInference(
            model=bittransgnn_model,
            dataset_name=dataset_name,
            graph_data=graph_data,
            joint_training=True,
            device=DEVICE,
            ext_cls_feats=None,          # None to force timing BERT extraction
            batch_size=batch_size,
            inductive=False,
            recompute_bert=True
        )

        # Stage A: one-time BERT CLS extraction timing
        bert_stage_stats = gcn_inf.time_update_cls_feats(warmup=0, repeat=1)
        print("[Stage A]", bert_stage_stats)

        # Stage B: per-epoch GCN full-graph timing (train+val+test)
        gcn_epoch_stats = gcn_joint_inf.time_gcn_full_graph_epoch(splits=("train","val","test"), warmup=1, repeat=5)
        print("[Stage B]", gcn_epoch_stats)

        # Stage C: per-epoch static GCN full-graph timing (train+val+test)
        gcn_static_epoch_stats = gcn_inf.time_gcn_full_graph_epoch_static(splits=("train","val","test"), warmup=1, repeat=5)
        print("[Stage C]", gcn_static_epoch_stats)

        # --- BitBERT (baseline) ---
        bert_inf = BitTransformerInference(
            model=bert_model,
            dataset_name=dataset_name,
            text_data=text_data,
            device=DEVICE
        )

        bert_epoch_stats = bert_inf.time_text_epoch(splits=("train","val","test"), warmup=1, repeat=5)
        print("[Transformer Baseline] epoch", bert_epoch_stats)

        # ====== Comet experiment ======
        exp = Experiment(
            api_key=API_KEY,
            workspace=WORKSPACE,
            project_name=PROJECT,
            auto_metric_logging=False,
            auto_param_logging=False,
            auto_output_logging="simple"
        )
        exp.set_name(f"{dataset_name}__{adj_type}")
        exp.add_tag(dataset_name); exp.add_tag(adj_type); exp.add_tag("runtime")

        # Log params (adjust as needed)
        exp.log_parameters({
            "dataset_name": dataset_name,
            "adj_type": adj_type,
            "device": DEVICE,
            "seed": SEED,
            "bert_pre_model": bert_pre_model,
            "batch_size": batch_size,
            "max_length": max_length,
            "joint_training": joint_training,
            "quantize_gcn": quantize_gcn,
            "gcn_num_states": gcn_num_states,
            "num_states": num_states,
            "num_bits_act": num_bits_act,
            "quantize_bert": quantize_bert,
            "quantize_embeddings": quantize_embeddings,
            "quantize_attention": quantize_attention,
            "lmbd": lmbd,
            "gcn_layers": gcn_layers,
            "n_hidden": n_hidden,
            "dropout": dropout,
            "regression": regression,
            "timing_warmup_stageA": WARMUP_A,
            "timing_repeat_stageA": REPEAT_A,
            "timing_warmup_stageBC": WARMUP_BC,
            "timing_repeat_stageBC": REPEAT_BC,
        })

        """
        # Optionally log graph scale if available
        for k in ("num_nodes", "num_edges", "nb_nodes", "nb_edges"):
            if hasattr(graph_data, k):
                exp.log_parameter(f"graph_{k}", getattr(graph_data, k))
        
        """

        # ====== Stage A: BERT CLS extraction over full set ======
        stats_A = gcn_inf.time_update_cls_feats(warmup=WARMUP_A, repeat=REPEAT_A)
        for k, v in stats_A.items():
            if k in ("what", "splits"): continue
            exp.log_metric(f"runtime/stageA_{k}", v)
        exp.log_other("runtime/stageA_what", stats_A.get("what", "stageA"))

        # ====== Stage B: GCN full-graph epoch (train+val+test) ======
        stats_B = gcn_joint_inf.time_gcn_full_graph_epoch(splits=("train","val","test"), warmup=WARMUP_BC, repeat=REPEAT_BC)
        for k, v in stats_B.items():
            if k in ("what", "splits"): continue
            exp.log_metric(f"runtime/stageB_{k}", v)
        exp.log_other("runtime/stageB_what", stats_B.get("what", "stageB"))
        exp.log_other("runtime/stageB_splits", ",".join(stats_B.get("splits", [])))

        # ====== Stage C: GCN static full-graph epoch (train+val+test) ======
        stats_C = gcn_inf.time_gcn_full_graph_epoch_static(splits=("train","val","test"), warmup=WARMUP_BC, repeat=REPEAT_BC)
        for k, v in stats_C.items():
            if k in ("what", "splits"): continue
            exp.log_metric(f"runtime/stageC_{k}", v)
        exp.log_other("runtime/stageC_what", stats_C.get("what", "stageC"))
        exp.log_other("runtime/stageC_splits", ",".join(stats_C.get("splits", [])))


        """
        # ====== Stage C1: Transformer epoch on *validation only* ======
        stats_C_val = bert_inf.time_text_epoch(splits=("val"), warmup=WARMUP_BC, repeat=REPEAT_BC)
        for k, v in stats_C_val.items():
            if k in ("what", "splits"): continue
            exp.log_metric(f"runtime/stageC_val/{k}", v)
        exp.log_other("runtime/stageC_val/what", stats_C_val.get("what", "stageC_val"))

        """

        # ====== Stage D (optional): Transformer epoch on full-set (train+val+test) ======
        stats_D_full = bert_inf.time_text_epoch(splits=("train","val","test"), warmup=WARMUP_BC, repeat=REPEAT_BC)
        for k, v in stats_D_full.items():
            if k in ("what", "splits"): continue
            exp.log_metric(f"runtime/stageD_full/{k}", v)
        exp.log_other("runtime/stageD_full/what", stats_D_full.get("what", "stageD_full"))
        exp.log_other("runtime/stageD_full/splits", ",".join(stats_D_full.get("splits", [])))

        # End the experiment (flush)
        exp.end()

print("All runs completed.")


