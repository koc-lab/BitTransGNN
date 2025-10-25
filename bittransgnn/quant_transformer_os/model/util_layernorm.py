import torch
from torch import nn
import torch.nn.functional as F
from quant_transformer_os.quantization import QuantizedModule, Quantizer


class QuantizedLayerNorm(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.layernorm = org_module
        if self.qoutput:
            self.layernorm_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, hidden_states, observation_mask=None):
        hidden_states = self.layernorm(hidden_states)
        if self.qoutput:
            hidden_states = self.layernorm_post_act_fake_quantize(hidden_states, observation_mask, 1)
        return hidden_states


class QuantizedSplitLayerNorm(QuantizedModule):

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.layernorm = nn.LayerNorm(org_module.normalized_shape, elementwise_affine=False)
        tmp = (org_module.bias.data / org_module.weight.data).detach().clone()
        self.bias = torch.nn.Parameter(tmp)
        if self.qoutput:
            self.layernorm_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, hidden_states, observation_mask=None):
        hidden_states = self.layernorm(hidden_states)
        hidden_states += self.bias
        if self.qoutput:
            hidden_states = self.layernorm_post_act_fake_quantize(hidden_states, observation_mask, 1)
        return hidden_states


class GammaResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul_gamma = False

    def set_gamma(self, gamma):
        self.mul_gamma = True
        self.gamma = torch.nn.Parameter(gamma.data.detach().clone())

    def forward(self, input, hidden_states):
        if self.mul_gamma:
            input = input * self.gamma
        return input + hidden_states


from types import SimpleNamespace

class QuantizedLayerNormPlus(nn.Module):

    def __init__(self, org_module):
        super(QuantizedLayerNormPlus, self).__init__()
        self.normalized_shape = org_module.normalized_shape
        self.eps = org_module.eps
        self.elementwise_affine = org_module.elementwise_affine
        self.weight = org_module.weight
        self.bias = org_module.bias

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.migrate = False
        self.migrate_scale = None

    def set_migrate(self, state):
        if self.migrate_scale is None:
            self.migrate = False
        else:
            self.migrate = state

    def set_migrate_scale(self, migrate_scale):
        self.migrate_scale = migrate_scale
        self.migrate = True

    def set_migrate_bias(self, migrate_bias):
        self.migrate_bias = migrate_bias
        self.migrate = True

    def forward(self, X):
        if self.migrate:
            X = X * self.migrate_scale + self.migrate_bias
        return X