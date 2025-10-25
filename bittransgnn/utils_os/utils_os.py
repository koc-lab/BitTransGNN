import torch
import torch.nn as nn
from quant_transformer_os.model.util_layernorm import QuantizedLayerNorm, QuantizedLayerNormPlus
from quant_transformer_os.quantization.quantized_module import QuantizedModule, QuantizedLayer, Quantizer, QLinear
from types import SimpleNamespace

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

def swap_ln_to_lnplus(module: nn.Module):
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
            swap_ln_to_lnplus(child)

        # Case B: plain nn.LayerNorm
        elif isinstance(child, nn.LayerNorm) and not isinstance(child, QuantizedLayerNormPlus):
            lnplus = QuantizedLayerNormPlus(child)   # <-- correct call
            # safely replace this child on the parent
            module._modules[name] = lnplus

        else:
            swap_ln_to_lnplus(child)


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

def dict_to_namespace(d):
    """
    Recursively convert a dictionary to a SimpleNamespace object
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

class DictNamespace(dict):
    """A dictionary that also supports attribute access and .get() method"""
    def __init__(self, d):
        super().__init__(d)
        for k, v in d.items():
            if isinstance(v, dict):
                self[k] = DictNamespace(v)
            else:
                self[k] = v
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value
