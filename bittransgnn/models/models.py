import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

import torch_geometric
from torch_geometric.nn import GCNConv

from quantization.binarize_model import quantize_tf_model
from quantization.binary_layers import BitLinear


class GCNConvTorch(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantize=False, num_states=2):
        super(GCNConvTorch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if quantize:
            # note that bias is not implemented for the quantized scenario
            self.lin = BitLinear(in_features, out_features, bias=False, num_states=num_states)
            self.register_parameter('bias', None)
        else:
            self.lin = nn.Linear(in_features, out_features, bias=False)
            if bias:
                #self.bias = nn.Parameter(torch.FloatTensor(out_features))
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        #stdv = 1. / math.sqrt(self.out_features)
        if self.bias is not None:
            #self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()
    
    def forward(self, input, adj):
        x = self.lin(input)
        output = torch.sparse.mm(adj, x)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNTorch(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout, activation=F.relu, quantize=False, num_states=2):
        super(GCNTorch, self).__init__()
        self.gcnconv1 = GCNConvTorch(in_features=in_features, out_features=hidden_features, quantize=quantize, num_states=num_states)
        self.gcnconv2 = GCNConvTorch(in_features=hidden_features, out_features=out_features, quantize=quantize, num_states=num_states)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gcnconv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcnconv2(x, adj)
        return x

class BitTransformer(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', nb_class=20, quantize=False, num_states=2, quantize_embeddings=False, num_bits_act=8.0, regression=False):
        super(BitTransformer, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        bert_model = AutoModel.from_pretrained(pretrained_model)
        if quantize or quantize_embeddings:
            self.bert_model = quantize_tf_model(bert_model, num_states=num_states, linear_quantize=quantize, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act)
        else:
            self.bert_model = bert_model
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        if quantize:
            self.classifier = BitLinear(in_features=self.feat_dim, out_features=nb_class, num_states=num_states, num_bits_act=num_bits_act)
        else:
            self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.regression = regression

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        if self.regression:
            pred = cls_logit
        else:
            pred = nn.Softmax(dim=1)(cls_logit)
            pred = torch.log(pred)
        return pred
    
class BitTransGNN(nn.Module):
    def __init__(self, pretrained_model='roberta-base', joint_training=True, quantize_gcn=False, gcn_num_states=2, nb_class=20, lmbd=0.7, gcn_layers=2, n_hidden=200, dropout=0.5, regression=False):
        super(BitTransGNN, self).__init__()
        self.lmbd = lmbd
        self.joint_training = joint_training
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCNTorch(in_features=self.feat_dim, hidden_features=n_hidden, out_features=nb_class, dropout=dropout, quantize=quantize_gcn, num_states=gcn_num_states)
        self.regression = regression

    def forward(self, graph_data, idx, temperature=1):
        if self.joint_training:
            input_ids, attention_mask = graph_data.input_ids[idx], graph_data.attention_mask[idx]
            if self.training:
                cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
                graph_data.cls_feats[idx] = cls_feats
            else:
                cls_feats = graph_data.cls_feats[idx]
        else:
            cls_feats = graph_data.cls_feats[idx]
        cls_logit = self.classifier(cls_feats)
        gcn_logit = self.gcn(x=graph_data.cls_feats, adj=graph_data.adj_sparse)[idx]
        if self.regression:
            gcn_pred, cls_pred = gcn_logit, cls_logit
            pred = gcn_pred * self.lmbd + cls_pred * (1 - self.lmbd)
        else:
            cls_pred = nn.Softmax(dim=1)(cls_logit/temperature)
            gcn_pred = nn.Softmax(dim=1)(gcn_logit/temperature)
            pred = (gcn_pred+1e-10) * self.lmbd + cls_pred * (1 - self.lmbd)
        return pred

class BitTransformerStudent(nn.Module):
    def __init__(self, pretrained_model='roberta-base', nb_class=20, regression=False):
        super(BitTransformerStudent, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.regression = regression

    def forward(self, graph_data, idx, temperature=1):
        input_ids, attention_mask = graph_data.input_ids[idx], graph_data.attention_mask[idx]
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        if self.regression:
            pred, soft_pred = cls_logit, cls_logit
        else:
            pred = nn.Softmax(dim=1)(cls_logit)
            pred = torch.log(pred)
            soft_pred = nn.Softmax(dim=1)(cls_logit/temperature)
            soft_pred = torch.log(soft_pred)
        return pred, soft_pred

