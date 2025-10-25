from typing import Optional
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from .binary_layers import BertBitSelfAttention, RobertaBitSelfAttention, BitLinear, BitEmbedding, BitSelfAttentionHF

# ---------- Linear/Embedding replacer ----------
def replace_layer(module, num_states: int,
                  linear_quantize: bool = True,
                  quantize_embeddings: bool = False,
                  num_bits_act: float = 8.0,
                  linear_backend: str = None,
                  embedding_backend: str = None):
    """
    Extended with optional backend selection for BitLinear/BitEmbedding.
    """
    if isinstance(module, nn.Linear):
        if linear_quantize:
            target_sd = deepcopy(module.state_dict())
            bias = module.bias is not None
            new_module = BitLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=bias,
                num_states=num_states,
                num_bits_act=num_bits_act,
            )
            new_module.load_state_dict(target_sd)
            if linear_backend and hasattr(new_module, "configure_inference_backend"):
                try:
                    new_module.configure_inference_backend(linear_backend)
                except Exception:
                    pass
            return new_module
        else:
            return module

    elif isinstance(module, nn.Embedding):
        if quantize_embeddings:
            target_sd = deepcopy(module.state_dict())
            new_module = BitEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                num_states=num_states,
            )
            new_module.load_state_dict(target_sd)
            if embedding_backend and hasattr(new_module, "configure_inference_backend"):
                try:
                    new_module.configure_inference_backend(embedding_backend)
                except Exception:
                    pass
            return new_module
        else:
            return module

    else:
        return module


def recursive_setattr(obj, attr, value):
    """
    Recursively sets the attribute of the object to the value being passed through the function.
    
    Args: 
        obj: the object being modified
        attr: the attribute being modified within the object
        value: the new value being assigned to the object's attribute
    """
    parts = attr.split('.', 1)
    if len(parts) == 1:
        setattr(obj, parts[0], value)
    else:
        recursive_setattr(getattr(obj, parts[0]), parts[1], value)

# ---------- Detect HF-style attention blocks ----------
def is_hf_self_attention_like(module: nn.Module) -> bool:
    """
    Conservative duck-typing check for HF Bert-like self-attention blocks.
    We only rely on attributes your BitSelfAttentionHF expects.
    """
    needed = [
        "query", "key", "value",
        "num_attention_heads", "attention_head_size", "all_head_size",
        "dropout", "transpose_for_scores"
    ]
    return all(hasattr(module, attr) for attr in needed)

# ---------- Replace a *single* module with BitSelfAttentionHF ----------
def replace_attention_layer_old(module: nn.Module,
                            num_bits_act: float = 8.0,
                            quantize_scores: bool = False,
                            quantize_probs: bool = False,
                            attn_backend: str = None) -> nn.Module:
    """
    If `module` looks like an HF attention block, wrap it with BitSelfAttentionHF.
    Otherwise return unchanged.
    Optionally set an inference backend if the wrapper supports it.
    """
    if is_hf_self_attention_like(module):
        wrapped = BitSelfAttentionHF(
            attn_module=module,
            num_bits_act=num_bits_act,
            quantize_scores=quantize_scores,
            quantize_probs=quantize_probs,
        )
        # Optional: pick a faster backend if available (e.g., "sdpa", "bnb-int8", "binary-emu", "fp32")
        if attn_backend and hasattr(wrapped, "configure_inference_backend"):
            try:
                wrapped.configure_inference_backend(attn_backend)
            except Exception:
                pass
        return wrapped
    return module

def replace_attention_layer(model, num_bits_act=None):
    if model.config.model_type == "bert":
        attention_model = BertBitSelfAttention
    elif model.config.model_type == "roberta":
        attention_model = RobertaBitSelfAttention
    else:
        raise NotImplementedError(f"Model type {model.config.model_type} not supported for attention replacement.")
    if num_bits_act is not None:
        model.config.num_bits_act = num_bits_act  # read by BitAttention at init
    
    for i, layer in enumerate(model.encoder.layer):
        old_attn = layer.attention.self
        new_attn = attention_model(
            model.config,
            position_embedding_type=getattr(old_attn, "position_embedding_type", "absolute"),
        )
        # Copy weights (Q/K/V/out) to keep initialization identical
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        layer.attention.self = new_attn
    return model

def quantize_tf_model(
    model: nn.Module,
    *,
    # pass 1 (per-layer)
    num_states: int = 2,
    linear_quantize: bool = True,
    quantize_embeddings: bool = True,
    num_bits_act: float = 8.0,
    linear_backend: str = None,
    embedding_backend: str = None,
    quantize_attention: bool = True,
    # pass 2 (attention wrapper)
    attn_num_bits_act: float = 8.0,
    quantize_scores: bool = False,
    quantize_probs: bool = False,
    attn_backend: str = None,
) -> nn.Module:
    # -------- pass 1: replace Linear / Embedding everywhere --------
    if quantize_attention:
        model.bert_model = replace_attention_layer(
            model.bert_model,
            num_bits_act=attn_num_bits_act,
        )

    for name, module in tuple(model.named_modules()):
        if not name:
            continue
        new_m = replace_layer(
            module,
            num_states=num_states,
            linear_quantize=linear_quantize,
            quantize_embeddings=quantize_embeddings,
            num_bits_act=num_bits_act,
            linear_backend=linear_backend,
            embedding_backend=embedding_backend,
        )
        if new_m is not module:
            recursive_setattr(model, name, new_m)

    """
    # -------- pass 2: wrap HF-style attention blocks --------
    if quantize_attention:
        for name, module in tuple(model.named_modules()):
            if not name:
                continue
            if is_hf_self_attention_like(module):
                wrapped = replace_attention_layer(
                    module,
                    num_bits_act=attn_num_bits_act,
                    #quantize_scores=quantize_scores,
                    #quantize_probs=quantize_probs,
                    #attn_backend=attn_backend,
                )
                if wrapped is not module:
                    recursive_setattr(model, name, wrapped)    
    """
    print("Quantization pass completed.")
    print(f"- Weight states: {num_states} | Act bits: {num_bits_act}")
    print(f"- Linear quant: {linear_quantize} ({linear_backend or 'default'})")
    print(f"- Embedding quant: {quantize_embeddings} ({embedding_backend or 'default'})")
    print(f"- Attention quant: {quantize_attention} ({attn_backend or 'default'})",
          f"| score_q: {quantize_scores} | prob_q: {quantize_probs}")

    return model

def quantize_bittransgnn_architecture(
    model: nn.Module,
    ckpt: dict,
    quantize_bert: bool,
    bert_quant_type: str,
    quantize_gcn: bool,
    num_states: int,
    joint_training: bool,
    lmbd: float,
    quantize_embeddings: bool,
    num_bits_act: float,
    quantize_attention: bool = True,
    *,
    # NEW (optional) attention + backends
    quantize_scores: bool = False,
    quantize_probs: bool = False,
    linear_backend: Optional[str] = None,       # "fp32", "cpu-int8-dynamic", "bnb-int8", "binary-emu"
    embedding_backend: Optional[str] = None,    # "fp32", "cpu-int8-dynamic"
    attn_backend: Optional[str] = None          # "fp32", "sdpa", "bnb-int8", "binary-emu"
):
    """
    Quantizes the BERTGCN model (BERT + classifier; GCN is handled in its own class).
    Compatible with attention swapping & backends.
    """
    if quantize_bert:
        print("model before quantization: ", model)
        if bert_quant_type == "QAT":
            # Quantize the architecture first, then load QAT-trained weights
            model = quantize_tf_model(
                model=model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            model.classifier = replace_layer(
                module=model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
            model.bert_model.load_state_dict(ckpt["bert_model"])
            model.classifier.load_state_dict(ckpt["classifier"])
        else:
            # PTQ: load FP weights, then quantize (typical)
            model.bert_model.load_state_dict(ckpt["bert_model"])
            model.classifier.load_state_dict(ckpt["classifier"])
            model = quantize_tf_model(
                model=model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            model.classifier = replace_layer(
                module=model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
    else:
        model.bert_model.load_state_dict(ckpt["bert_model"])
        model.classifier.load_state_dict(ckpt["classifier"])

    if quantize_gcn:
        print("A quantized GCN model is being used.")

    if quantize_bert or quantize_gcn:
        print("model after quantization: ", model)
    else:
        print("model: ", model)

    if joint_training:
        print(f"BERT and GCN jointly trained; outputs interpolated by λ={lmbd}.")
        print("BERT cls logits are interpolated with GCN output for final prediction.")
    else:
        print("GCN is trained; BERT provides features for GCN.")
        print(f"BERT cls logits are interpolated with GCN output by λ={lmbd} for final prediction.")
    return model


def quantize_bittransformer_for_inference(
    model: nn.Module,
    ckpt: dict,
    joint_training: bool,
    quantize_bert: bool,
    num_states: int,
    bitbert_quant_type: str,
    quantize_embeddings: bool,
    num_bits_act: float,
    quantize_attention: bool = True,
    *,
    quantize_scores: bool = False,
    quantize_probs: bool = False,
    linear_backend: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    attn_backend: Optional[str] = None
):
    """
    BitBERT-only (BERT + classifier) inference quantization; attention & backends compatible.
    """
    if quantize_bert:
        if (not joint_training) and bitbert_quant_type == "PTQ":
            # Load FP then quantize
            model.bert_model.load_state_dict(ckpt["bert_model"])
            model.classifier.load_state_dict(ckpt["classifier"])
            model = quantize_tf_model(
                model=model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            model.classifier = replace_layer(
                module=model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
        else:
            # Quantize structure first (e.g., QAT use-case), then load trained weights
            model = quantize_tf_model(
                model=model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            model.classifier = replace_layer(
                module=model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
            model.bert_model.load_state_dict(ckpt["bert_model"])
            model.classifier.load_state_dict(ckpt["classifier"])
    else:
        model.bert_model.load_state_dict(ckpt["bert_model"])
        model.classifier.load_state_dict(ckpt["classifier"])
    return model


def quantize_bittransgnn_for_inference(
    model: nn.Module,
    ckpt: dict,
    joint_training: bool,
    quantize_bert: bool,
    num_states: int,
    bitbert_quant_type: str,
    quantize_embeddings: bool,
    num_bits_act: float,
    quantize_attention: bool = True,
    *,
    quantize_scores: bool = False,
    quantize_probs: bool = False,
    linear_backend: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    attn_backend: Optional[str] = None
):
    """
    BitBERT+GCN inference quantization; attention & backends compatible.
    """
    if quantize_bert:
        if (not joint_training) and bitbert_quant_type == "PTQ":
            model.bert_model.load_state_dict(ckpt["bert_model"])
            model.classifier.load_state_dict(ckpt["classifier"])
            model = quantize_tf_model(
                model=model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            model.classifier = replace_layer(
                module=model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
        else:
            model = quantize_tf_model(
                model=model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            model.classifier = replace_layer(
                module=model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
            model.bert_model.load_state_dict(ckpt["bert_model"])
            model.classifier.load_state_dict(ckpt["classifier"])
    else:
        model.bert_model.load_state_dict(ckpt["bert_model"])
        model.classifier.load_state_dict(ckpt["classifier"])

    model.gcn.load_state_dict(ckpt["gcn"])
    return model


def quantize_teacher_architecture(
    teacher_model: nn.Module,
    teacher_ckpt: dict,
    joint_training: bool,
    quantize_teacher_bert: bool,
    teacher_bert_quant_type: str,
    teacher_num_states: int,
    quantize_teacher_embeddings: bool,
    num_bits_act: float,
    quantize_attention: bool = True,
    *,
    quantize_scores: bool = False,
    quantize_probs: bool = False,
    linear_backend: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    attn_backend: Optional[str] = None
):
    """
    Teacher model quantization with attention/backends compatibility.
    """
    if quantize_teacher_bert:
        if (not joint_training) and teacher_bert_quant_type == "PTQ":
            teacher_model.bert_model.load_state_dict(teacher_ckpt["bert_model"])
            teacher_model.classifier.load_state_dict(teacher_ckpt["classifier"])
            teacher_model.gcn.load_state_dict(teacher_ckpt["gcn"])
            teacher_model = quantize_tf_model(
                model=teacher_model,
                num_states=teacher_num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_teacher_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            teacher_model.classifier = replace_layer(
                module=teacher_model.classifier,
                num_states=teacher_num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
        else:
            teacher_model = quantize_tf_model(
                model=teacher_model,
                num_states=teacher_num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_teacher_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            teacher_model.classifier = replace_layer(
                module=teacher_model.classifier,
                num_states=teacher_num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
            teacher_model.bert_model.load_state_dict(teacher_ckpt["bert_model"])
            teacher_model.classifier.load_state_dict(teacher_ckpt["classifier"])
            teacher_model.gcn.load_state_dict(teacher_ckpt["gcn"])
    else:
        teacher_model.bert_model.load_state_dict(teacher_ckpt["bert_model"])
        teacher_model.classifier.load_state_dict(teacher_ckpt["classifier"])
        teacher_model.gcn.load_state_dict(teacher_ckpt["gcn"])
    return teacher_model


def quantize_student_architecture(
    student_model: nn.Module,
    student_ckpt: dict,
    quantize_student_bert: bool,
    student_bert_quant_type: str,
    num_states: int,
    quantize_embeddings: bool,
    num_bits_act: float,
    quantize_attention: bool = True,
    *,
    quantize_scores: bool = False,
    quantize_probs: bool = False,
    linear_backend: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    attn_backend: Optional[str] = None
):
    """
    Student model quantization with attention/backends compatibility.
    """
    if quantize_student_bert:
        if student_bert_quant_type == "QAT":
            student_model = quantize_tf_model(
                model=student_model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            student_model.classifier = replace_layer(
                module=student_model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
            student_model.bert_model.load_state_dict(student_ckpt["bert_model"])
            student_model.classifier.load_state_dict(student_ckpt["classifier"])
        else:
            # PTQ order: load FP → quantize
            student_model.bert_model.load_state_dict(student_ckpt["bert_model"])
            student_model.classifier.load_state_dict(student_ckpt["classifier"])
            student_model = quantize_tf_model(
                model=student_model,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=quantize_embeddings,
                num_bits_act=num_bits_act,
                quantize_attention=quantize_attention,
                attn_num_bits_act=num_bits_act,
                quantize_scores=quantize_scores,
                quantize_probs=quantize_probs,
                linear_backend=linear_backend,
                embedding_backend=embedding_backend,
                attn_backend=attn_backend,
            )
            student_model.classifier = replace_layer(
                module=student_model.classifier,
                num_states=num_states,
                linear_quantize=True,
                quantize_embeddings=False,
                num_bits_act=num_bits_act,
                linear_backend=linear_backend,
                embedding_backend=None,
            )
    else:
        student_model.bert_model.load_state_dict(student_ckpt["bert_model"])
        student_model.classifier.load_state_dict(student_ckpt["classifier"])
    return student_model
