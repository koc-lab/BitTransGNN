from torch import nn
import torch.nn.functional as F
from torch import Tensor
import sys
from typing import Optional

def activation_quant(x: Tensor, num_bits=8.0):
    """
    Per token quantization.

    Args: 
        x: input tensor
        num_bits: num_bits used to quantize the original tensor. 
        If num_bits not in [1.0, 1.58, 2.0, 2.32, 8.0], we automatically use 8-bit quantization for the activations.
    """

    if num_bits == 1.0:
        scale = x.abs().mean()
        e = x.mean()
        y = (x - e).sign() * scale
    elif num_bits == 1.58:
        scale = 1.0 / x.abs().mean().clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-1, 1) / scale
    elif num_bits == 2.0:
        scale = 1.0 / x.abs().mean().clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-2, 1) / scale
    elif num_bits == 2.32:
        scale = 1.0 / x.abs().mean().clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-2, 2) / scale
    elif num_bits == 8.0:
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
    else:
        #we also quantize any invalid entry for num_bits to 8 bits
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
    return y, scale

def two_bit_centered_map_func(x):
    """
    Quantization procedure for the case when a centered, 2-bit quantization is preferred.
    The original activations are mapped to the set {-2, -1, 1, 2} in this case.
    Due to low performance, this quantization was not preferred in the experiments.

    Args: 
        x: input tensor
    """
    dtype = x.dtype
    x = x - x.mean()
    scale = x.abs().mean()
    pos_high = x > scale
    pos_low_1 = (x < scale)
    pos_low_0 = x > 0
    pos_high_val = 2 * (pos_high).to(dtype=dtype)
    pos_low_val = (pos_low_0 * pos_low_1).to(dtype=dtype)
    neg_high = x < (-scale)
    neg_low_1 = x > (-scale)
    neg_low_0 = x < 0
    neg_high_val = -2 * (neg_high).to(dtype=dtype)
    neg_low_val = (neg_low_0 * neg_low_1).to(dtype=dtype)
    quantized_x = pos_high_val + pos_low_val + neg_high_val + neg_low_val
    y = quantized_x * scale
    return y

def weight_quant(w: Tensor, num_states: int = 2, centered: Optional[bool] = False):
    """
    Weight quantization.

    Args: 
        w: weight tensor with shape [d, k]
        num_states: integer to determine the quantization set. Must be within the set {2, 3, 4, 5}.
        centered: boolean variable to determine whether to use a centered quantized representation for the case when 
        num_states = 4. Note that for the rest of the scenarios, the quantization set is already centered.
    """
    if num_states == 2:
        # weight quantization scheme for the 1-bit scenario
        scale = w.abs().mean()
        e = w.mean()
        u = (w - e).sign() * scale
    elif num_states == 3:
        #Per-tensor quantization to 1.58 bits.
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
    elif num_states == 4:
        # weight quantization scheme for the 2-bit scenario
        if centered:
            u = two_bit_centered_map_func(w)
        else:
            scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
            u = (w * scale).round().clamp_(-2, 1) / scale
    elif num_states == 5:
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-2, 2) / scale
    else:
        sys.exit(f"Entered number of quantization states (num_states={num_states}) was not implemented, please try a different quantization rate.")
    return u

class BitLinear(nn.Linear):
    """
    Linear layer with bit quantization.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: boolean variable to determine whether an additive bias term will be learned by the model.
            Default: False
        num_states: number of states used to quantize the weights of the linear layer
            Default: 2
        num_bits_act: number of bits used to quantize the activations
            Default: 8.0

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, num_states: int = 2, num_bits_act: float = 8.0):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.num_states = num_states
        self.num_bits_act = num_bits_act

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x: input tensor.
        """
        num_states = self.num_states
        num_bits_act = self.num_bits_act
        w = self.weight

        # STE using detach
        if num_bits_act == 32.0:
            x_quant = x
        else:
            x_quant, _ = activation_quant(x, num_bits_act)
        x_quant = x + (x_quant - x).detach()
        w_quant = w + (weight_quant(w, num_states) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

class BitEmbedding(nn.Embedding):
    """
    Embedding layer with bit quantization. 
    Note that only the weights of the embedding layer are quantized, and the sampling from the dictionary is maintained full-precision.

    Args:
        num_embeddings: size of the dictionary of embeddings
        embedding_dim: size of each embedding vector
        num_states: number of states used to quantize the weights of the embedding layer

    """

    def __init__(self, num_embeddings: int, embedding_dim: int, num_states: int = 2):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.num_states = num_states

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of BitEmbedding.

        Args:
            x: input embedding indices.
        """
        num_states = self.num_states
        w = self.weight

        # STE using detach
        w_quant = w + (weight_quant(w, num_states) - w).detach()
        y = F.embedding(x, w_quant, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return y