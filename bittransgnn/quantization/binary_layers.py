import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import sys
from typing import Optional, Tuple
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

def activation_quant(x: Tensor, num_bits=8.0):
    """
    Per token quantization.

    Args: 
        x: input tensor
        num_bits: num_bits used to quantize the original tensor. 
        If num_bits not in [1.0, 1.58, 2.0, 2.32, 8.0], we automatically use 8-bit quantization for the activations.
    """

    if num_bits == 1.0:
        e     = x.mean(dim=-1, keepdim=True)
        scale = x.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        y     = (x - e).sign() * scale
    elif num_bits == 1.58:
        scale = 1.0 / x.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-1, 1) / scale
    elif num_bits == 2.0:
        scale = 1.0 / x.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-2, 1) / scale
    elif num_bits == 2.32:
        scale = 1.0 / x.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-2, 2) / scale
    elif num_bits == 8.0:
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
    else:
        #we also quantize any invalid entry for num_bits to 8 bits
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
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
        e = w.mean(dim=1, keepdim=True)
        scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        u = (w - e).sign() * scale
    elif num_states == 3:
        #Per-tensor quantization to 1.58 bits.
        scale = 1.0 / w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
    elif num_states == 4:
        # 4 levels including zero: integers in {-2,-1,0,1} (centered around 0)
        # (kept for continuity with your design; if you prefer 4 symmetric nonzero levels,
        #  use integers in {-3,-1,1,3} and rescale accordingly.)
        scale = 1.0 / w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
        if centered:
            # same as below; retained 'centered' flag for API compatibility
            u = (w * scale).round().clamp_(-2, 1) / scale
        else:
            u = (w * scale).round().clamp_(-2, 1) / scale
    elif num_states == 5:
        scale = 1.0 / w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-2, 2) / scale
    else:
        raise ValueError(f"Entered number of quantization states (num_states={num_states}) was not implemented, please try a different quantization rate.")
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

        if num_bits_act == 32.0:
            x_quant = x
        else:
            # STE using detach
            x_quant = activation_quant(x, num_bits_act)

        w_quant = weight_quant(w, num_states)
        
        if self.training:
            x_quant = x + (x_quant - x).detach()
            w_quant = w + (w_quant - w).detach()

        y = F.linear(x_quant, w_quant, bias=self.bias)
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

        w_quant = weight_quant(w, num_states)

        # STE using detach
        if self.training:
            w_quant = w + (w_quant - w).detach()
        
        y = F.embedding(x, w_quant, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return y

class BertBitSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.num_bits_act = getattr(config, "num_bits_act", None)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.num_bits_act:
            if self.training:
                query_layer = query_layer + (activation_quant(query_layer, self.num_bits_act) - query_layer).detach()
                key_layer = key_layer + (activation_quant(key_layer, self.num_bits_act) - key_layer).detach()
                value_layer = value_layer + (activation_quant(value_layer, self.num_bits_act) - value_layer).detach()
            else:
                query_layer = activation_quant(query_layer, self.num_bits_act)
                key_layer = activation_quant(key_layer, self.num_bits_act)
                value_layer = activation_quant(value_layer, self.num_bits_act)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class RobertaBitSelfAttention(RobertaSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.num_bits_act = getattr(config, "num_bits_act", None)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.num_bits_act:
            if self.training:
                query_layer = query_layer + (activation_quant(query_layer, self.num_bits_act) - query_layer).detach()
                key_layer = key_layer + (activation_quant(key_layer, self.num_bits_act) - key_layer).detach()
                value_layer = value_layer + (activation_quant(value_layer, self.num_bits_act) - value_layer).detach()
            else:
                query_layer = activation_quant(query_layer, self.num_bits_act)
                key_layer = activation_quant(key_layer, self.num_bits_act)
                value_layer = activation_quant(value_layer, self.num_bits_act)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BitSelfAttentionHF(nn.Module):
    def __init__(self, attn_module: nn.Module, num_bits_act: float = 8.0,
                 quantize_scores: bool = False, quantize_probs: bool = False):
        super().__init__()
        self.attn = attn_module
        self.num_bits_act = float(num_bits_act)
        self.quantize_scores = bool(quantize_scores)
        self.quantize_probs = bool(quantize_probs)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        is_cross = encoder_hidden_states is not None
        key_value_states = encoder_hidden_states if is_cross else hidden_states

        # 1) Linear projections (your BitLinear handles weight/act quant in its own forward)
        Q = self.attn.query(hidden_states)
        K = self.attn.key(key_value_states)
        V = self.attn.value(key_value_states)

        # 2) [B, H, T, Dh]
        Q = self.attn.transpose_for_scores(Q)
        K = self.attn.transpose_for_scores(K)
        V = self.attn.transpose_for_scores(V)

        training = self.training
        if self.num_bits_act != 32.0:
            if self.training:
                Q = Q + (activation_quant(Q, self.num_bits_act) - Q).detach()
                K = K + (activation_quant(K, self.num_bits_act) - K).detach()
                V = V + (activation_quant(V, self.num_bits_act) - V).detach()
            else:
                Q = activation_quant(Q, self.num_bits_act)
                K = activation_quant(K, self.num_bits_act)
                V = activation_quant(V, self.num_bits_act)

        Dh = self.attn.attention_head_size
        scale = 1.0 / math.sqrt(Dh)

        # "fp32" baseline (your original path)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
        if self.quantize_scores and self.num_bits_act != 32.0:
            if self.training:
                scores = scores + (activation_quant(scores, self.num_bits_act) - scores).detach()
            else:
                scores = activation_quant(scores, self.num_bits_act)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        probs = self.attn.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask
        if self.quantize_probs and self.num_bits_act != 32.0:
            if self.training:
                probs = probs + (activation_quant(probs, self.num_bits_act) - probs).detach()
            else:
                probs = activation_quant(probs, self.num_bits_act)
        context = torch.matmul(probs, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.attn.all_head_size)
        return (context, probs) if output_attentions else (context,)
