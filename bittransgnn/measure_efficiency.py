# Minimal efficiency benchmark for BitTransformer / BitTransGNN
# with real INT8 kernels (CPU: torch.ao.dynamic, CUDA: bitsandbytes) and a backend report.

import copy
import yaml
from pathlib import Path
import argparse, json, math, time
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

# ===== Your project imports (keep your paths) =====
from data.loader.dataloaders import TextDataObject, GraphDataObject
from models import BitTransformer, BitTransGNN, GCNConvLayerTorch
from quantization.binarize_model import quantize_tf_model
from quantization.binary_layers import BitLinear, BitSelfAttentionHF, BitEmbedding  # optional

# ================== BENCHMARK CORE ==================
import time, threading, re, contextlib, gc
import torch
import torch.nn as nn

# ---- NVML power sampler (board power) ----
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

def _parse_cuda_index(device_str: str) -> int:
    m = re.match(r"cuda:(\d+)", str(device_str).lower())
    return int(m.group(1)) if m else 0

class NvmlPowerSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.01):
        self.device_index = device_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self._samples = []
        self._handle = None

    def start(self):
        if not _HAS_NVML:
            return
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
                self._samples.append((t, mw/1000.0))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop_and_summarize(self):
        if not _HAS_NVML or self._thread is None:
            return {"energy_j": None, "avg_watts": None, "peak_watts": None,
                    "samples": 0, "duration_s": None}
        self._stop.set()
        self._thread.join(timeout=1.0)
        samples = self._samples
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        if len(samples) < 2:
            return {"energy_j": 0.0, "avg_watts": None, "peak_watts": None,
                    "samples": len(samples), "duration_s": 0.0}
        # trapezoidal integration
        E = 0.0
        for (t0,w0),(t1,w1) in zip(samples[:-1], samples[1:]):
            dt = max(0.0, t1-t0)
            E += 0.5*(w0+w1)*dt
        dur = samples[-1][0]-samples[0][0]
        return {"energy_j": E, "avg_watts": E/dur if dur>0 else None,
                "peak_watts": max(w for _,w in samples),
                "samples": len(samples), "duration_s": dur}

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _time_ms(fn, device):
    _sync(device); t0 = time.perf_counter()
    fn()
    _sync(device); return (time.perf_counter()-t0)*1000.0

# ---- CUDA-events batch timer (accurate on GPU) ----
def time_with_cuda_events(forward_fn, device: torch.device, iters: int = 50, warmup: int = 10):
    if device.type != "cuda":
        # fall back to wall-clock if on CPU
        _sync(device); t0 = time.perf_counter()
        for _ in range(warmup): forward_fn()
        _sync(device); t0 = time.perf_counter()
        for _ in range(iters): forward_fn()
        _sync(device); return (time.perf_counter() - t0) * 1000.0 / max(1, iters)

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        forward_fn()
        torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)

    total_ms = 0.0
    for _ in range(iters):
        start.record()
        forward_fn()
        end.record()
        torch.cuda.synchronize(device)
        total_ms += start.elapsed_time(end)  # milliseconds
    return total_ms / max(1, iters)  # avg ms per batch


def estimate_epoch_time_loader_window(model_step, loader, device: torch.device, K: int = 32, warmup: int = 4):
    import itertools
    # warmup (trigger caching, collation paths, etc.)
    it = iter(loader)
    for _ in range(min(warmup, K)):
        try:
            b = next(it)
        except StopIteration:
            return None
        model_step(b)

    it = iter(loader)
    _sync(device); t0 = time.perf_counter()
    seen = 0
    for b in itertools.islice(it, K):
        model_step(b)
        seen += 1
    _sync(device)
    if seen == 0:
        return None
    avg_ms = (time.perf_counter() - t0) * 1000.0 / seen
    return (avg_ms / 1000.0) * len(loader)  # seconds

# ---- main benchmark runner ----
def benchmark_module(
    *,
    forward_fn, # callable that executes ONE forward step (already closes over inputs/model)
    device: torch.device,
    iters: int = 50,
    warmup: int = 10,
    name: str = "",
    samples_per_iter: int = None,
    get_ops_dict=None,
    use_profiler: bool = False,
    profile_with_stack: bool = False,
    power_sample: bool = False,
    device_str: str = None,
):
    """
    OOM-safe benchmark runner that matches your builder pattern:
      - Runs forward_fn() under inference_mode()
      - Optional: NVML power sampling
      - Optional: torch.profiler (kept lightweight)
      - Returns a dict with latency, peak mem, throughput, (and power if enabled),
        plus anything returned by get_ops_dict()
    """
    import gc, time
    from contextlib import nullcontext

    torch.set_grad_enabled(False)

    # -------- helpers --------
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # NVML power sampler setup
    sampler = None
    if power_sample and device.type == "cuda":
        try:
            idx = _parse_cuda_index(device_str or str(device))
            sampler = NvmlPowerSampler(device_index=idx, interval_s=0.01)
        except Exception:
            sampler = None

    prof_ctx = nullcontext()
    if use_profiler:
        try:
            import torch.profiler as tprof
            acts = [tprof.ProfilerActivity.CPU]
            if device.type == "cuda":
                acts.append(tprof.ProfilerActivity.CUDA)
            active_steps = min(5, max(1, iters // 10))
            prof_ctx = tprof.profile(
                activities=acts,
                schedule=tprof.schedule(wait=1, warmup=1, active=active_steps, repeat=1),
                record_shapes=False,
                profile_memory=False,
                with_stack=profile_with_stack,
                on_trace_ready=None,
            )
        except Exception:
            prof_ctx = nullcontext()

    # -------- warmup --------
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync()
    if sampler is not None:
        sampler.start()

    try:
        # Warmup (outside profiler to keep it light)
        for _ in range(max(0, warmup)):
            _ = forward_fn()
        _sync()

        # Timed run (optionally under profiler)
        t0 = time.perf_counter()
        with prof_ctx:
            for i in range(iters):
                _ = forward_fn()
                _sync()
                if use_profiler and hasattr(prof_ctx, "step"):
                    prof_ctx.step()
        total_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        # Power sampler summary
        power_stats = {}
        if sampler is not None:
            try:
                power_stats = sampler.stop_and_summarize()
            except Exception:
                power_stats = {}

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------- stats --------
    avg_ms = total_ms / max(1, iters)
    out = {
        "name": name or "benchmark",
        "avg_latency_ms": avg_ms,
        "iters": iters,
        "device": str(device),
    }
    if samples_per_iter:
        out["samples_per_iter"] = int(samples_per_iter)
        out["throughput_items_per_s"] = (samples_per_iter * 1000.0) / avg_ms

    if device.type == "cuda":
        try:
            out["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            pass

    if power_stats:
        out.update(power_stats)  # adds energy_j, avg_watts, peak_watts, samples, duration_s

    if get_ops_dict is not None:
        try:
            out.update(get_ops_dict() or {})
        except Exception as e:
            out["ops_error"] = str(e)

    return out

def _now(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()

def _elapsed_ms(t0, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0

def profile_module(module: nn.Module, inputs: tuple, iters: int = 50, warmup: int = 10,
                   get_ops_fn=None):
    """
    Times module(*inputs). If get_ops_fn is provided, it should be a callable returning a dict of ops.
    """
    device = None
    for x in inputs:
        if isinstance(x, torch.Tensor):
            device = x.device
            break
    device = device or torch.device("cpu")

    with torch.no_grad():
        for _ in range(warmup):
            out = module(*inputs)
            if isinstance(out, tuple):
                _ = out[0] if isinstance(out[0], torch.Tensor) else None

    t0 = _now(device)
    with torch.no_grad():
        for _ in range(iters):
            out = module(*inputs)
            if isinstance(out, tuple):
                _ = out[0] if isinstance(out[0], torch.Tensor) else None
    total_ms = _elapsed_ms(t0, device)
    avg_ms = total_ms / iters

    stats = {"avg_latency_ms": avg_ms, "iters": iters}
    if get_ops_fn is not None:
        try:
            ops = get_ops_fn()
            if isinstance(ops, dict):
                stats.update(ops)
        except Exception as e:
            stats["ops_error"] = str(e)
    return stats

def apply_cpu_dynamic_int8_linear(model: nn.Module) -> nn.Module:
    """Replace nn.Linear with dynamic-quant int8 linears on CPU."""
    from torch.ao.quantization import quantize_dynamic
    qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)
    return qmodel

def apply_cpu_dynamic_int8_embedding(model: nn.Module) -> nn.Module:
    """Replace nn.Embedding with dynamic-quant int8 embeddings on CPU."""
    from torch.ao.quantization import quantize_dynamic
    qmodel = quantize_dynamic(model, {nn.Embedding}, dtype=torch.quint8, inplace=False)
    return qmodel

def make_embeddings_contiguous_(model: nn.Module):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                # force the underlying storage to be contiguous
                m.weight.set_(m.weight.contiguous())

def swap_linears_to_bnb(module: nn.Module) -> nn.Module:
    """
    In-place: recursively replace nn.Linear with bitsandbytes Linear8bitLt.
    Leaves already-swapped linears untouched. Returns the same module.
    """
    try:
        import bitsandbytes as bnb
    except Exception as e:
        raise RuntimeError("[swap_linears_to_bnb] bitsandbytes not available") from e

    bnb_types = (getattr(bnb.nn, "Linear8bitLt", tuple()),)
    bnb_types = tuple(t for t in bnb_types if t)  # filter None

    for name, child in list(module.named_children()):
        # If it's already a bnb linear, recurse but don't replace
        if bnb_types and isinstance(child, bnb_types):
            swap_linears_to_bnb(child)
            continue

        if isinstance(child, nn.Linear):
            # Build replacement with same shape & bias flag
            new_m = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                has_fp16_weights=False,
            ).to(child.weight.device)
            new_m.eval()
            with torch.no_grad():
                new_m.weight.copy_(child.weight.data)
                if child.bias is not None:
                    new_m.bias.copy_(child.bias.data)

            module._modules[name] = new_m
        else:
            swap_linears_to_bnb(child)

    return module

def report_backends(model: nn.Module):
    n_lin = n_bnb = n_dyn = n_bit = 0
    # detect backend types
    try:
        import bitsandbytes as bnb
        BNBLinear = getattr(bnb.nn, "Linear8bitLt", tuple())
    except Exception:
        BNBLinear = tuple()

    try:
        from torch.ao.nn.quantized.dynamic import Linear as AOQDynLinear
    except Exception:
        AOQDynLinear = tuple()

    for _, m in model.named_modules():
        if BNBLinear and isinstance(m, BNBLinear):
            n_bnb += 1
        elif AOQDynLinear and isinstance(m, AOQDynLinear):
            n_dyn += 1
        elif isinstance(m, nn.Linear):
            n_lin += 1
        try:
            if isinstance(m, BitLinear): n_bit += 1
        except Exception:
            pass

    print(f"[report] Linear(fp): {n_lin} | bnb: {n_bnb} | torch.ao dyn: {n_dyn} | BitLinear: {n_bit}")

class MiniCounter:
    def __init__(self, act_bits: float = 32.0, w_bits: float = 32.0):
        self.macs_linear = 0
        self.macs_qk = 0
        self.macs_pv = 0
        # bitmults (proxy for precision-aware work)
        self.bitmults_linear = 0.0
        self.bitmults_attn_qk = 0.0
        self.bitmults_attn_pv = 0.0
        self.act_bits = act_bits
        self.w_bits = w_bits   # used for Linears; attention uses act_bits for both operands

    @property
    def macs_total(self): return self.macs_linear + self.macs_qk + self.macs_pv
    @property
    def flops_total(self): return 2 * self.macs_total

    def as_dict(self):
        return {
            "macs_linear": int(self.macs_linear),
            "macs_attn_qk": int(self.macs_qk),
            "macs_attn_pv": int(self.macs_pv),
            "macs_total": int(self.macs_total),
            "flops_total": int(self.flops_total),
            "bitmults_linear": int(self.bitmults_linear),
            "bitmults_attn_qk": int(self.bitmults_attn_qk),
            "bitmults_attn_pv": int(self.bitmults_attn_pv),
            "bitmults_total": int(self.bitmults_linear + self.bitmults_attn_qk + self.bitmults_attn_pv),
        }

def attach_mini_counters(model: nn.Module, act_bits: float, w_bits: float):
    C = MiniCounter(act_bits=act_bits, w_bits=w_bits)
    hooks = []

    def linear_hook(mod: nn.Linear, inp, out):
        x = inp[0]
        in_f = x.shape[-1]
        rows = x.numel() // in_f
        macs = rows * in_f * mod.out_features
        C.macs_linear += macs
        wb = C.w_bits
        if hasattr(mod, "num_states") and mod.num_states:
            try:
                wb = math.log2(float(mod.num_states))
            except Exception:
                pass
        C.bitmults_linear += macs * (wb * C.act_bits)

    def attn_hook(mod, inp, out):
        hidden = inp[0]                     # [B, T, Hd]
        B, T = int(hidden.shape[0]), int(hidden.shape[1])
        core = getattr(mod, "attn", mod)    # BitSelfAttentionHF has .attn
        H  = int(getattr(core, "num_attention_heads", 12))
        Dh = int(getattr(core, "attention_head_size", hidden.shape[-1] // H))
        mac_qk = B * H * (T * T * Dh)
        mac_pv = B * H * (T * T * Dh)
        C.macs_qk += mac_qk
        C.macs_pv += mac_pv
        C.bitmults_attn_qk += mac_qk * (C.act_bits * C.act_bits)   # Q and K bitwidths
        C.bitmults_attn_pv += mac_pv * (C.act_bits * C.act_bits)   # probs usually FP, but we keep simple

    for m in model.modules():
        if isinstance(m, (nn.Linear, BitLinear)):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (BitSelfAttentionHF,)):
            hooks.append(m.register_forward_hook(attn_hook))
        elif all(hasattr(m, a) for a in ("query","key","value","num_attention_heads","attention_head_size")):
            hooks.append(m.register_forward_hook(attn_hook))
    return C, hooks

def attach_gcn_torch_counters(model: nn.Module):
    """
    Counts MACs for sparse propagation in your GCNConvLayerTorch:
      SpMM(adj @ X): ~ nnz(adj) * F_out
    (Dense Linear inside the layer is already covered by your Linear hooks.)
    Returns (counter, hooks)
    """
    class GCNTorchCounter:
        def __init__(self):
            self.macs_gcn_spmm = 0
            self.adds_gcn_bias = 0  # optional bias adds
        @property
        def macs_gcn_total(self):
            return self.macs_gcn_spmm  # adds are not MACs; keep separate
        def as_dict(self):
            return {
                "macs_gcn_spmm": int(self.macs_gcn_spmm),
                "adds_gcn_bias": int(self.adds_gcn_bias),
                "macs_gcn_total": int(self.macs_gcn_total),
            }

    C = GCNTorchCounter()
    hooks = []

    def _nnz_sparse(A: torch.Tensor) -> int:
        try:
            return int(A._nnz())
        except Exception:
            pass
        try:
            return int(A.values().numel())
        except Exception:
            pass
        return int((A != 0).sum().item())

    GCNClass = None
    try:
        GCNClass = GCNConvLayerTorch
    except Exception:
        GCNClass = None

    def is_our_gcn_layer(m):
        if GCNClass is not None:
            return isinstance(m, GCNClass)
        return m.__class__.__name__ == "GCNConvLayerTorch"

    def gcn_hook(mod, inp, out):
        # forward(self, input, adj) -> output
        # inp: (input, adj)
        if not inp or len(inp) < 2:
            return
        adj = inp[1]
        if not torch.is_tensor(adj):
            return
        # nnz(adj)
        E = _nnz_sparse(adj)
        # out_features from the inner Linear:
        try:
            Fout = int(mod.lin.out_features)
        except Exception:
            # fallback from actual output shape
            try:
                Fout = int(out.shape[-1])
            except Exception:
                return
        # MACs for SpMM
        C.macs_gcn_spmm += E * Fout

        # optional: bias adds (one add per node per out_feature)
        try:
            if getattr(mod, "bias", None) is not None:
                N = int(out.shape[0])  # rows of output
                C.adds_gcn_bias += N * Fout
        except Exception:
            pass

    # attach to all instances of the layer
    for m in model.modules():
        if is_our_gcn_layer(m):
            hooks.append(m.register_forward_hook(gcn_hook))

    return C, hooks

# Scenarios (fp32 / int8 / bnb-int8 / binary)
def scenarios_for_device(device: torch.device, use_bnb: bool):
    lin_bnb = "bnb-int8" if (device.type=="cuda" and use_bnb) else "fp32"
    return [
        # 32-32-32-32
        {"name": "fp32_32-32-32-32",
         "W_bits": 32, "E_bits": 32, "Act_bits": 32,
         "linear_backend": "fp32", "embedding_backend": "fp32", "attn_backend": "fp32"},
        # 1-1-1-32  (still FP attention)
        {"name": "bin_1-1-1-32_fp",
         "W_bits": 1, "E_bits": 1, "Act_bits": 32,
         "linear_backend": "fp32", "embedding_backend": "fp32", "attn_backend": "fp32"},
        {"name": "bin_1-1-1-8_int8Linear",
         "W_bits": 1, "E_bits": 1, "Act_bits": 8,
         "linear_backend": ("cpu-int8-dynamic" if device.type=="cpu" else lin_bnb),
         #"embedding_backend": ("cpu-int8-dynamic" if device.type=="cpu" else "fp32"), 
         "embedding_backend": "fp32",
         "attn_backend": "fp32"},
        {"name": "bin_1-1-1-1_fp",
         "W_bits": 1, "E_bits": 1, "Act_bits": 1,
         "linear_backend": ("cpu-int8-dynamic" if device.type=="cpu" else lin_bnb), 
         "embedding_backend": "fp32", "attn_backend": "fp32"},
        # 1-1-1-8 (use dynamic int8 on CPU or bnb-int8 on CUDA for linears)
        # 1-1-1-1 (binary emu)
    ]

def build_bert_from_config(cfg: Dict[str, Any], use_bitlinear=False):
    exp, mod, params, load = cfg["experiment_configs"], cfg["model_configs"], cfg["parameters"], cfg["load_configs"]
    device         = torch.device(exp["device"])
    dataset_name   = exp["dataset_name"]
    bert_pre_model = exp["bert_pre_model"]
    seed           = exp["seed"]
    max_length     = mod["max_length"]
    batch_size     = params["batch_size"]
    torch.manual_seed(seed)

    text_data = TextDataObject(dataset_name, batch_size, adj_type=mod.get("adj_type", None), seed=seed)
    nb_class = text_data.nb_class
    regression = (dataset_name.lower() == "stsb")
    model = BitTransformer(pretrained_model=bert_pre_model, nb_class=nb_class, regression=regression).to(device).eval()

    if use_bitlinear:
        model = quantize_tf_model(
            model=model, num_states=mod["num_states"],
            linear_quantize=mod["quantize_bert"], quantize_embeddings=mod["quantize_embeddings"], num_bits_act=mod["num_bits_act"],
            linear_backend=mod["linear_backend"],
            embedding_backend=mod["embedding_backend"], 
            quantize_attention=mod["quantize_attention"],
            attn_num_bits_act=mod["num_bits_act"],
            attn_backend=mod["attn_backend"]
        ).to(device).eval()

    text_data.set_dataloaders_bert(model, max_length)
    eval_loader = text_data.loaders["val"]
    return model, eval_loader, device

def build_bertgcn_from_config(cfg: Dict[str, Any], use_bitlinear=False):
    exp, mod, params, load = cfg["experiment_configs"], cfg["model_configs"], cfg["parameters"], cfg["load_configs"]
    device         = torch.device(exp["device"])
    dataset_name   = exp["dataset_name"]
    bert_pre_model = exp["bert_pre_model"]
    seed           = exp["seed"]
    max_length     = mod["max_length"]
    batch_size     = params["batch_size"]
    inference_type = exp["inference_type"]

    train_only = True if inference_type == "inductive" else False

    torch.manual_seed(seed)
    graph_data = GraphDataObject(dataset_name, batch_size, adj_type=mod["adj_type"], seed=seed, train_only=train_only)
    nb_class = graph_data.nb_class
    regression = (dataset_name.lower() == "stsb")

    lmbd = params["lmbd"]

    model = BitTransGNN(
        pretrained_model=bert_pre_model, joint_training=params["joint_training"], 
        quantize_gcn=mod["quantize_gcn"], gcn_num_states=mod["gcn_num_states"],
        nb_class=nb_class, lmbd=lmbd, gcn_layers=params["gcn_layers"], n_hidden=params["graph_hidden_size"],
        dropout=params["dropout"], regression=regression
    ).to(device).eval()
    graph_data.set_transformer_data(model, max_length)
    graph_data.set_graph_data(model)

    if use_bitlinear:

        model = quantize_tf_model(
            model=model, num_states=mod["num_states"],
            linear_quantize=mod["quantize_bert"], quantize_embeddings=mod["quantize_embeddings"], num_bits_act=mod["num_bits_act"],
            linear_backend=mod["linear_backend"],
            embedding_backend=mod["embedding_backend"],
            quantize_attention=mod["quantize_attention"],
            attn_num_bits_act=mod["num_bits_act"],
            attn_backend=mod["attn_backend"]
        ).to(device).eval()

    
    eval_loader = graph_data.get_dataloaders_bertgcn()[0]["val"]
    return model, eval_loader, graph_data, device

def first_batch_bert(eval_loader, device):
    it = iter(eval_loader)
    b = next(it)
    if isinstance(b, dict):
        ids = b["input_ids"].to(device)
        mask = b.get("attention_mask", torch.ones_like(ids)).to(device)
        labels = b.get("labels", torch.zeros(ids.size(0), dtype=torch.long, device=device))
        return (ids, mask, labels)
    else:
        return tuple(t.to(device) if torch.is_tensor(t) else t for t in b)

def first_batch_bertgcn(eval_loader, device):
    it = iter(eval_loader)
    b = next(it)
    if isinstance(b, tuple):
        return tuple(t.to(device) if torch.is_tensor(t) else t for t in b)
    elif isinstance(b, dict):
        return tuple(v.to(device) if torch.is_tensor(v) else v for v in b.values())
    else:
        return (b,)

def _model_step_bert(model, batch, device: torch.device):
    if isinstance(batch, dict):
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch.get("attention_mask", torch.ones_like(ids)).to(device, non_blocking=True)
        _ = model(input_ids=ids, attention_mask=mask)
    else:
        ids, mask = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        _ = model(input_ids=ids, attention_mask=mask)

def apply_scenario_to_config(cfg: Dict[str, Any], sc: Dict[str, Any]) -> Dict[str, Any]:
    cfg2 = json.loads(json.dumps(cfg))  # deep copy
    cfg2["model_configs"]["quantize_bert"]      = True
    cfg2["model_configs"]["quantize_attention"] = True
    cfg2["model_configs"]["num_bits_act"]       = sc["Act_bits"]
    cfg2["model_configs"]["quantize_embeddings"]= (sc["E_bits"] != 32)
    if sc["W_bits"] == 1:
        cfg2["model_configs"]["num_states"] = 2
    cfg2["model_configs"]["linear_backend"]    = sc["linear_backend"]
    cfg2["model_configs"]["embedding_backend"] = sc["embedding_backend"]
    cfg2["model_configs"]["attn_backend"]      = sc["attn_backend"]
    return cfg2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--which", choices=["bert", "bertgcn", "bertgcn_d", "bertgcn (doc_doc)", "bertgcn_d (doc_doc)", "all"], default="all")
    ap.add_argument("--config_yaml_bitbert", default=Path(__file__).parent.parent.joinpath("methods/bitbert/configs/inference_config.yaml"))
    ap.add_argument("--config_yaml_bittransgnn", default=Path(__file__).parent.parent.joinpath("methods/bittransgnn/configs/inference_config.yaml"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--use_bnb_int8", action="store_true")
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument("--save_csv", type=str, default=None)
    args = ap.parse_args()

    cfg_sample = yaml.load(open(args.config_yaml_bitbert), Loader=yaml.FullLoader)
    #dataset_name = cfg_sample["experiment_configs"]["dataset_name"]
    dataset = args.dataset
    if dataset == "all":
        dataset_name_list = ["mrpc", "20ng", "cola", "ohsumed"]
    else:
        dataset_name_list = [dataset]
    
    for dataset_name in dataset_name_list:
        cfg_sample["experiment_configs"]["dataset_name"] = dataset_name
        print("dataset_name: ", dataset_name)

        # load the needed configs depending on --which
        to_run = []
        if args.which in ("bert", "all"):
            with open(args.config_yaml_bitbert) as f:
                cfg_bert = yaml.load(f, Loader=yaml.FullLoader)
            cfg_bert["experiment_configs"]["device"] = args.device
            to_run.append(("bert", cfg_bert))
            
        if args.which in ("bertgcn", "all"):
            with open(args.config_yaml_bittransgnn) as f:
                cfg_bgcn = yaml.load(f, Loader=yaml.FullLoader)
            cfg_bgcn["experiment_configs"]["device"] = args.device
            to_run.append(("bertgcn", cfg_bgcn))
            if args.which == "all":
                cfg_bgcn_docdoc = copy.deepcopy(cfg_bgcn)
                cfg_bgcn_docdoc["model_configs"]["adj_type"] = "doc_doc"
                to_run.append(("bertgcn (doc_doc)", cfg_bgcn_docdoc))

        elif args.which == "bertgcn (doc_doc)":
            with open(args.config_yaml_bittransgnn) as f:
                cfg_bgcn = yaml.load(f, Loader=yaml.FullLoader)
            cfg_bgcn["experiment_configs"]["device"] = args.device
            cfg_bgcn["model_configs"]["adj_type"] = "doc_doc"
            to_run.append(("bertgcn (doc_doc)", cfg_bgcn))

        if args.which in ("bertgcn_d", "all"):
            with open(args.config_yaml_bittransgnn) as f:
                cfg_bgcn_d = yaml.load(f, Loader=yaml.FullLoader)
            cfg_bgcn_d["experiment_configs"]["device"] = args.device
            to_run.append(("bertgcn_d", cfg_bgcn_d))
            if args.which == "all":
                cfg_bgcn_d_docdoc = copy.deepcopy(cfg_bgcn_d)
                cfg_bgcn_d_docdoc["model_configs"]["adj_type"] = "doc_doc"
                to_run.append(("bertgcn_d (doc_doc)", cfg_bgcn_d_docdoc))

        elif args.which == "bertgcn_d (doc_doc)":
            with open(args.config_yaml_bittransgnn) as f:
                cfg_bgcn_d = yaml.load(f, Loader=yaml.FullLoader)
            cfg_bgcn_d["experiment_configs"]["device"] = args.device
            cfg_bgcn_d["model_configs"]["adj_type"] = "doc_doc"
            to_run.append(("bertgcn_d (doc_doc)", cfg_bgcn_d))


        device = torch.device(args.device)
        all_results = []

        # loop over requested model types
        for which, cfg in to_run:
            cfg["experiment_configs"]["dataset_name"] = dataset_name
            if which == "bert":
                base_model, base_loader, _device = build_bert_from_config(cfg)
                get_first = lambda loader: first_batch_bert(loader, _device)
                builder = build_bert_from_config
                graph_data = None
            else:
                base_model, base_loader, graph_data, _device = build_bertgcn_from_config(cfg)
                get_first = lambda loader: first_batch_bertgcn(loader, _device)
                builder = build_bertgcn_from_config

            print(f"\n=== benchmarking: {which} on {args.device} ===")

            for sc in scenarios_for_device(device, args.use_bnb_int8):
                if sc["name"] == "bin_1-1-1-1_binary":
                    use_bitlinear = True
                else:
                    use_bitlinear = False
                use_bitlinear = False # delete later
                cfg2 = apply_scenario_to_config(cfg, sc)
                built = builder(cfg2, use_bitlinear=use_bitlinear)
                # unpack per-builder
                if which == "bert":
                    model, eval_loader, _device = built
                else:
                    model, eval_loader, graph_data, _device = built

                if sc["embedding_backend"] == "cpu-int8-dynamic":
                    make_embeddings_contiguous_(model)

                # ---- REAL KERNEL SWAPS (post-quantize) ----
                if sc["linear_backend"] == "cpu-int8-dynamic" and device.type == "cpu":
                    if which == "bert":
                        model = apply_cpu_dynamic_int8_linear(model)
                    else:
                        model.bert_model = apply_cpu_dynamic_int8_linear(model.bert_model)
                        model.classifier = apply_cpu_dynamic_int8_linear(model.classifier)
                elif sc["linear_backend"] == "bnb-int8" and device.type == "cuda":
                    if which == "bert": 
                        model = swap_linears_to_bnb(model.bert_model)
                    else:
                        model.bert_model = swap_linears_to_bnb(model.bert_model)
                        model.classifier = swap_linears_to_bnb(model.classifier)

                if sc["embedding_backend"] == "cpu-int8-dynamic" and device.type == "cpu":
                    model = apply_cpu_dynamic_int8_embedding(model)

                # backend report
                report_backends(model)

                # capture one batch
                batch = get_first(eval_loader)

                # counters
                counter, hooks = attach_mini_counters(model, act_bits=float(sc["Act_bits"]), w_bits=float(sc["W_bits"]))
                gcn_counter, gcn_hooks = attach_gcn_torch_counters(model)

                def one_step():
                    if which == "bert":
                        ids, mask, labels = batch
                        if sc["embedding_backend"] == "cpu-int8-dynamic" and device.type == "cpu":
                            ids  = ids.to("cpu", non_blocking=True).contiguous().long()
                            mask = mask.to("cpu", non_blocking=True).contiguous()
                        _ = model(input_ids=ids, attention_mask=mask)
                    else:
                        recompute_bert = True if which == "bertgcn_d" else False
                        _ = model(graph_data.convert_device(_device), batch[0][0].to(_device), recompute_bert=recompute_bert)

                # 3) run benchmark (profiler + NVML)
                stats_hardware = benchmark_module(
                    forward_fn=one_step,
                    device=device,
                    iters=args.iters,
                    warmup=args.warmup,
                    samples_per_iter=batch[0].shape[0] if which=="bert" else None,  # batch size for throughput (optional)
                    get_ops_dict=lambda: {
                        **counter.as_dict(),
                        **gcn_counter.as_dict(),
                    },
                    use_profiler=True,
                    profile_with_stack=False,
                    power_sample=True,
                    device_str=args.device,
                )

                # ---------- Epoch-time estimates for fairness (BERT only) ----------
                num_batches = len(eval_loader)

                print("num_batches:", num_batches)

                # (A) CUDA-events per-batch timing × #batches -> full-dataset estimate
                avg_ms_evt = time_with_cuda_events(
                    forward_fn=one_step,
                    device=device,
                    iters=args.iters,
                    warmup=args.warmup
                )
                est_epoch_s_batches = (avg_ms_evt / 1000.0) * float(num_batches)
                est_epoch_ms_batches = (avg_ms_evt) * float(num_batches)

                # ---------- Epoch-level estimates & energy scaling ----------
                # Per-iter wall time (ms)
                #per_iter_s = (stats_hardware["avg_latency_ms"] / 1000.0)
                per_iter_ms = (stats_hardware["avg_latency_ms"])
                # Energy per iter (J) if NVML was active
                per_iter_energy_j = None
                if "energy_j" in stats_hardware and stats_hardware["energy_j"] is not None:
                    per_iter_energy_j = stats_hardware["energy_j"] / max(1, args.iters)

                #num_batches = len(eval_loader)

                # Defaults
                epoch_time_ms = None
                epoch_energy_j = None

                if which == "bert":
                    epoch_time_ms = est_epoch_ms_batches
                    if per_iter_energy_j is not None:
                        epoch_energy_j = per_iter_energy_j * num_batches

                elif which == "bertgcn":
                    # one GCN full-graph pass per epoch
                    epoch_time_ms = per_iter_ms
                    if per_iter_energy_j is not None:
                        epoch_energy_j = per_iter_energy_j  # x1

                elif which == "bertgcn_d":
                    # GCN full-graph pass *after every batch*
                    epoch_time_ms = per_iter_ms * num_batches
                    if per_iter_energy_j is not None:
                        epoch_energy_j = per_iter_energy_j * num_batches

                # (Optional) If you want a single joint epoch estimate for the standard BERTGCN:
                #   joint_epoch_time_s = (BERT epoch from the BERT run) + (this GCN epoch)
                # You can compute it outside this loop by pairing results later, or store partials here.

                class Runner(nn.Module):
                    def forward(self):
                        if which == "bert":
                            ids, mask, labels = batch
                            out = model(input_ids=ids, attention_mask=mask)
                            return out[1] if isinstance(out, (tuple, list)) else out
                        else:    
                            recompute_bert = True if which == "bertgcn_d" else False
                            idx = batch[0][0].to(_device)
                            out = model(graph_data.convert_device(_device), idx, recompute_bert=recompute_bert)
                            if isinstance(out, dict) and "logits" in out: return out["logits"]
                            if isinstance(out, (tuple, list)): return out[2]
                            return out

                runner = Runner().to(device).eval()
                stats = profile_module(runner, inputs=tuple(), iters=args.iters, warmup=args.warmup,
                                    get_ops_fn=lambda: counter.as_dict())
                stats.update(gcn_counter.as_dict())
                stats.update(stats_hardware)

                for h in hooks: h.remove()
                for h in gcn_hooks: h.remove()

                row = {
                    "model": which,
                    "scenario": sc["name"],
                    "adj_type": cfg["model_configs"]["adj_type"],
                    **stats,
                    "avg_latency_ms_events": avg_ms_evt,
                    "num_batches": num_batches,
                    "bert_est_epoch_s_batches": est_epoch_ms_batches,
                    "**num_batches**": num_batches,
                    "**epoch_time_ms**": epoch_time_ms,
                    "**epoch_energy_j**": epoch_energy_j,
                }
                print(f"\n== {which} :: {sc['name']} ==")
                for k, v in row.items():
                    print(f"{k:>20}: {v}")
                all_results.append(row)

        with open(Path(__file__).parent / f"benchmark_results/benchmark_results_{dataset_name}_{device}.txt", "w") as f:
            print("path: ")
            print(Path(__file__).parent / f"benchmark_results/benchmark_results_{dataset_name}_{device}.txt")
            for result in all_results:
                for k, v in result.items():
                    f.write(f"{k:>20}: {v}\n")
            print("Results saved!")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[saved] {args.save_json}")

    if args.save_csv:
        import csv
        keys = ["model", "scenario", "adj_type", "avg_latency_ms", "iters",
                "macs_total", "flops_total", "macs_linear", "macs_attn_qk", "macs_attn_pv",
                "bitmults_total", "bitmults_linear", "bitmults_attn_qk", "bitmults_attn_pv",
                "avg_latency_ms_events", "num_batches", "bert_est_epoch_s_batches",
                "**num_batches**", "**epoch_time_ms**", "**epoch_energy_j**",]
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_results:
                w.writerow({k: r.get(k) for k in keys})
        print(f"[saved] {args.save_csv}")


if __name__ == "__main__":
    main()
