import yaml
import os
from easydict import EasyDict
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import torch
import torch.nn as nn

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    #"20ng": ("sentence", None), 
    #"mr": ("sentence", None),
    #"ohsumed": ("sentence", None),
    "20ng": ("sentence", None),
    "mr": ("sentence", None),
    "ohsumed": ("sentence", None),
    "R8": ("sentence", None),
    "R52": ("sentence", None)
}

"""
class BertClassifier(nn.Module):
    def __init__(self, pretrained_model='roberta-base', nb_class=20, regression=False):
        super(BertClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.regression = regression

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        cls_feats = self.bert_model(input_ids, attention_mask, token_type_ids=token_type_ids)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        logits = {"cls_logit": cls_logit.clone()}
        if self.regression:
            pred = cls_logit
        else:
            pred = nn.Softmax(dim=1)(cls_logit)
            pred = torch.log(pred)
        #return pred, logits
        return logits
"""    

class BertForSeqClsNoPooler(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # make the pooler a no-op (so we don't mistakenly use it)
        #self.bert.pooler = nn.Identity()
        self.pooler = nn.Identity()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        cls = out[0][:, 0]  # match your training
        logits = self.classifier(cls)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def load_dataset_labels(config_data):
    if config_data.task_name in ['20ng', 'mr', 'ohsumed', 'R8', 'R52', 'cola', 'mrpc', 'stsb', 'rte']:
        return load_custom_dataset_labels(config_data)
    # else, load from GLUE
    # datasets
    raw_datasets = load_dataset("glue", config_data.task_name, revision="bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
    # num_labels
    if config_data.is_regression:
        num_labels = 1
        label_list = None
    else:
        label_list = raw_datasets['train'].features['label'].names
        num_labels = len(label_list)
    return raw_datasets, num_labels, label_list


def load_model_org(config_model, config_data, num_labels):
    # num_labels first to indentity the classification heads
    tokenizer = AutoTokenizer.from_pretrained(
        config_model.tokenizer_name if config_model.tokenizer_name else config_model.model_name_or_path,
        cache_dir=config_model.cache_dir,
        use_fast=config_model.use_fast_tokenizer,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    config_tmp = AutoConfig.from_pretrained(
        config_model.config_name if config_model.config_name else config_model.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=config_data.task_name,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    if hasattr(config_model, 'attn_dropout'):
        config_tmp.attention_probs_dropout_prob = config_model.attn_dropout
    if hasattr(config_model, 'hidden_dropout'):
        config_tmp.hidden_dropout_prob = config_model.hidden_dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        config_model.model_name_or_path,
        from_tf=bool(".ckpt" in config_model.model_name_or_path),
        config=config_tmp,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    return tokenizer, model


def load_model(config_model, config_data, num_labels):
    # 1) Tokenizer from base model (since your folder has no tokenizer files)
    tokenizer = AutoTokenizer.from_pretrained(
        config_model.tokenizer_name or "bert-base-uncased",
        cache_dir=config_model.cache_dir,
        use_fast=config_model.use_fast_tokenizer,
        revision=config_model.model_revision,
        use_auth_token=True if getattr(config_model, "use_auth_token", False) else None,
    )

    # 2) Config from base model
    config_tmp = AutoConfig.from_pretrained(
        config_model.config_name or "bert-base-uncased",
        num_labels=num_labels,
        finetuning_task=config_data.task_name,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if getattr(config_model, "use_auth_token", False) else None,
    )
    if hasattr(config_model, "attn_dropout"):
        config_tmp.attention_probs_dropout_prob = config_model.attn_dropout
    if hasattr(config_model, "hidden_dropout"):
        config_tmp.hidden_dropout_prob = config_model.hidden_dropout

    # 3) Fresh model from config
    #model = AutoModelForSequenceClassification.from_config(config_tmp)
    model = BertForSeqClsNoPooler(config_tmp)
    #model = BertClassifier(pretrained_model="bert-base-uncased", nb_class=num_labels, regression=config_data.is_regression)
    #model.config = config_tmp
    #model.num_labels = num_labels

    # 4) Load your state dict
    #    Expect something like: /path/to/folder/checkpoint.pth
    ckpt_dir = config_model.model_name_or_path  # your folder with checkpoint.pth
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    sd = torch.load(ckpt_path, map_location="cpu")

    print("ckpt_dir: ", ckpt_dir)

    # 1) BERT backbone
    if "bert_model" in sd and isinstance(sd["bert_model"], dict):
        bert_sd = sd["bert_model"]                # keys like "embeddings.word_embeddings.weight"
    else:
        # monolithic state_dict with prefixes like "bert_model.xxx"
        bert_sd = { k[len("bert_model."):]: v for k, v in sd.items() if k.startswith("bert_model.") }

    # Optional: strip DDP "module." if present inside the sub-dict
    bert_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in bert_sd.items() }

    missing, unexpected = model.bert.load_state_dict(bert_sd, strict=False)
    print("bert load: missing", missing, "unexpected", unexpected)

    # 2) Classifier head
    if "classifier" in sd and isinstance(sd["classifier"], dict):
        cls_sd = sd["classifier"]                 # keys like "weight", "bias"
    else:
        cls_sd = { k[len("classifier."):]: v for k, v in sd.items() if k.startswith("classifier.") }

    missing_cls, unexpected_cls = model.classifier.load_state_dict(cls_sd, strict=False)
    print("classifier load: missing", missing_cls, "unexpected", unexpected_cls)

    #print("sd", sd)
    #print("model", model)
    #print("model keys", model.state_dict().keys())
    #print("sd keys", sd.keys())

    #model.bert.load_state_dict(sd["bert_model"], strict=False)
    #model.classifier.load_state_dict(sd["classifier"], strict=False)
    #model.bert_model.load_state_dict(sd["bert_model"], strict=False)
    #model.bert.load_state_dict(sd["bert_model"], strict=False)
    #model.classifier.load_state_dict(sd["classifier"], strict=False)

    # 3) Load encoder weights
    #missing_bert, unexpected_bert = model.bert.load_state_dict(sd["bert_model"], strict=False)
    #print("bert load: missing", missing_bert, "unexpected", unexpected_bert)

    # 4) Load classifier weights
    #missing_cls, unexpected_cls = model.classifier.load_state_dict(sd["classifier"], strict=False)
    #print("classifier load: missing", missing_cls, "unexpected", unexpected_cls)

    return tokenizer, model

def load_model_alt(config_model, config_data, num_labels):
    # 1) Tokenizer from base model (since your folder has no tokenizer files)
    tokenizer = AutoTokenizer.from_pretrained(
        config_model.tokenizer_name or "bert-base-uncased",
        cache_dir=config_model.cache_dir,
        use_fast=config_model.use_fast_tokenizer,
        revision=config_model.model_revision,
        use_auth_token=True if getattr(config_model, "use_auth_token", False) else None,
    )

    # 2) Config from base model
    config_tmp = AutoConfig.from_pretrained(
        config_model.config_name or "bert-base-uncased",
        num_labels=num_labels,
        finetuning_task=config_data.task_name,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if getattr(config_model, "use_auth_token", False) else None,
    )
    if hasattr(config_model, "attn_dropout"):
        config_tmp.attention_probs_dropout_prob = config_model.attn_dropout
    if hasattr(config_model, "hidden_dropout"):
        config_tmp.hidden_dropout_prob = config_model.hidden_dropout

    # 3) Fresh model from config
    model = BertClassifier(pretrained_model="bert-base-uncased", nb_class=num_labels, regression=config_data.is_regression)
    model.config = config_tmp

    # 4) Load your state dict
    #    Expect something like: /path/to/folder/checkpoint.pth
    ckpt_dir = config_model.model_name_or_path  # your folder with checkpoint.pth
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    sd = torch.load(ckpt_path, map_location="cpu")

    #print("sd", sd)
    #print("model", model)
    #print("model keys", model.state_dict().keys())
    #print("sd keys", sd.keys())

    model.bert_model.load_state_dict(sd["bert_model"], strict=False)
    model.classifier.load_state_dict(sd["classifier"], strict=False)

    # 3) Load encoder weights
    #missing_bert, unexpected_bert = model.bert.load_state_dict(sd["bert_model"], strict=False)
    #print("bert load: missing", missing_bert, "unexpected", unexpected_bert)

    # 4) Load classifier weights
    #missing_cls, unexpected_cls = model.classifier.load_state_dict(sd["classifier"], strict=False)
    #print("classifier load: missing", missing_cls, "unexpected", unexpected_cls)

    return tokenizer, model


def preprocess_dataset(config_data, training_args, raw_datasets, label_to_id, tokenizer):
    #print(raw_datasets["train"].column_names)    # tokenize the data
    #print(raw_datasets.column_names)  # For a single Dataset object
    sentence1_key, sentence2_key = task_to_keys[config_data.task_name]
    if config_data.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False
    max_seq_length = config_data.max_seq_length

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if config_data.task_name not in ["20ng", "mr", "ohsumed", "R8", "R52"]:
            print("config_data.task_name", config_data.task_name)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            else:
                print("No label_to_id provided, and labels are present in the dataset.")
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not config_data.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return raw_datasets


def check_return_data(raw_datasets, data_type, do_type, max_samples):
    if not do_type:
        return None
    if do_type and data_type not in raw_datasets:
        raise ValueError(f'do- {data_type} requires a {data_type} dataset')
    type_dataset = raw_datasets[data_type]
    if max_samples is not None:
        type_dataset = type_dataset.shuffle().select(range(max_samples))
    return type_dataset

import numpy as np
import scipy.sparse as sp
import pickle as pkl

SEP_WORD = "<<SEP>>"
PAIR_DATASETS = ["mrpc", "rte", "stsb"]
BASE = "/auto/k2/aykut4/kumbasar/1_bit_llm/bittransgnnv2/bittransgnn/dataset"
BASE_PAIR = "/auto/k2/aykut4/kumbasar/1_bit_llm/bittransgnnv2/bittransgnn/dataset_paired"

def split_ab(line: str, sep: str = SEP_WORD):
    line = line.strip()
    if not line:
        return "", None
    if sep in line:
        a, b = line.split(sep, 1)
        return a.strip(), b.strip()
    return line, None  # single-sentence datasets

def parse_index_file(filename):
    """
    Parses the index file given by the filename.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """
    Creates a mask based over the indices based on the labels introduced.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_custom_dataset_labels(config_data):
    """
    Loads input corpus from ./dataset directory based on dataset_name.

    ind.dataset_name.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_name.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_name.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_name.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_name.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_name.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_name.ally => the labels for instances in ind.dataset_name.allx as numpy.ndarray object;
    ind.dataset_name.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_name.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_name: Dataset name
    :param adj_type: None (full), no-doc (ww+wd), no-ww(dd+wd), doc-doc (only dd)
    :return: All data input files loaded (as well the training/test data).
    """
    dataset_name = config_data.task_name
    print("dataset_name: ", dataset_name)
    if dataset_name in PAIR_DATASETS:
        base_path = BASE_PAIR
    else:
        base_path = BASE
    names = ['y', 'ty', 'ally']
    objects = []
    for i in range(len(names)):
        with open(f"{base_path}/ind.{dataset_name}.{names[i]}", 'rb') as f:
            objects.append(pkl.load(f))

    y, ty, ally = tuple(objects)
    print(y.shape, ty.shape, ally.shape)

    train_idx_orig = parse_index_file(
        f"{base_path}/{dataset_name}.train.index")
    train_size = len(train_idx_orig)
    print("train_size: ", train_size)

    labels = np.vstack((ally, ty))

    print(len(labels))

    val_size = train_size - y.shape[0]
    test_size = ty.shape[0]
    print("val_size: ", val_size)
    print("test_size: ", test_size)

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    
    idx_test = range(ally.shape[0], ally.shape[0] + test_size)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print("y_train.shape, y_val.shape, y_test.shape")
    print(y_train.shape, y_val.shape, y_test.shape)

    nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    nb_labels = y_train.shape[1]
    # transform one-hot label to class ID for pytorch computation
    if dataset_name == "stsb":
        y = y_train + y_val + y_test
        y = torch.from_numpy(y).to(dtype=torch.float32)
    else:
        y = torch.LongTensor((y_train + y_val +y_test).argmax(axis=1))
    label = {}
    #label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]
    label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]
    print("label['train']")
    print(label['train'])
    print("label['val']")
    print(label['val'])
    print("label['test']")
    print(label['test'])

    # Load shuffled doc names and cleaned text
    with open(f"{base_path}/{dataset_name}_shuffle.txt") as f_meta, \
         open(f"{base_path}/corpus/{dataset_name}_shuffle.txt") as f_text:
        meta_lines = [line.strip() for line in f_meta if line.strip()]
        text_lines = [line.strip() for line in f_text if line.strip()]

    # Build train/validation/test splits
    train_text_lines = text_lines[:nb_train]
    val_text_lines = text_lines[nb_train:nb_train+nb_val]
    test_text_lines = text_lines[-nb_test:] if nb_test > 0 else text_lines[:0]
    assert len(train_text_lines) == nb_train
    assert len(val_text_lines) == nb_val
    assert len(test_text_lines) == nb_test

    if dataset_name in PAIR_DATASETS:
        # 2) Split into (A,B) per line using <<SEP>>
        a_list, b_list = [], []
        for line in text_lines:
            a, b = split_ab(line)   # b is None for single-sentence datasets
            a_list.append(a)
            b_list.append(b)
            train_lines_a = a_list[:nb_train]
            train_lines_b = b_list[:nb_train]
            val_lines_a = a_list[nb_train:nb_train+nb_val]
            val_lines_b = b_list[nb_train:nb_train+nb_val]
            test_lines_a = a_list[-nb_test:] if nb_test > 0 else a_list[:0]
            test_lines_b = b_list[-nb_test:] if nb_test > 0 else b_list[:0]
        assert len(train_lines_a) == nb_train
        assert len(train_lines_b) == nb_train
        assert len(val_lines_a) == nb_val
        assert len(val_lines_b) == nb_val
        assert len(test_lines_a) == nb_test
        assert len(test_lines_b) == nb_test

        train_data = [{"sentence1": train_lines_a[i], "sentence2": train_lines_b[i], "label": int(label['train'][i])} for i in range(nb_train)]
        val_data = [{"sentence1": val_lines_a[i], "sentence2": val_lines_b[i], "label": int(label['val'][i])} for i in range(nb_val)]
        test_data = [{"sentence1": test_lines_a[i], "sentence2": test_lines_b[i], "label": int(label['test'][i])} for i in range(nb_test)]
    else:
        train_lines = text_lines[:nb_train]
        val_lines = text_lines[nb_train:nb_train+nb_val]
        test_lines = text_lines[-nb_test:] if nb_test > 0 else text_lines
        assert len(train_lines) == nb_train
        assert len(val_lines) == nb_val
        assert len(test_lines) == nb_test
        train_data = [{"sentence": train_lines[i], "label": int(label['train'][i])} for i in range(nb_train)]
        val_data = [{"sentence": val_lines[i], "label": int(label['val'][i])} for i in range(nb_val)]
        test_data = [{"sentence": test_lines[i], "label": int(label['test'][i])} for i in range(nb_test)]

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(train_data)),
        "validation": Dataset.from_pandas(pd.DataFrame(val_data)),
        "test": Dataset.from_pandas(pd.DataFrame(test_data))
    })

    #label_list = None
    label_list = [str(i) for i in range(nb_labels)]

    return raw_datasets, nb_labels, label_list

"""
def set_dataloaders_bert(self, model, max_length):
    batch_size = self.batch_size
    nb_train = self.nb_train
    nb_val = self.nb_val
    nb_test = self.nb_test
    label = self.label
    dataset_name = self.dataset_name
    # load documents and compute input encodings
    if dataset_name in PAIR_DATASETS:
        base_path = BASE_PAIR
    else:
        base_path = BASE
    print("base_path: ", base_path)
    corpus_file = f'{base_path}/corpus/{dataset_name}_shuffle.txt'
    with open(corpus_file, 'r') as f:
        text = f.read().replace('\\', '').split('\n')

    # 2) Split into (A,B) per line using <<SEP>>
    a_list, b_list = [], []
    for line in text:
        a, b = split_ab(line)   # b is None for single-sentence datasets
        a_list.append(a)
        b_list.append(b)

    if self.dataset_name in PAIR_DATASETS:
        # 3) Tokenize in pair mode (tokenizers handle None in b_list)
        enc = model.tokenizer(
            a_list,
            b_list,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_token_type_ids=True  # BERT uses these; RoBERTa will return zeros
        )
    else:
        enc = model.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_token_type_ids=True  # BERT uses these; RoBERTa will return zeros
        )

    input_ids_all = enc["input_ids"]
    attn_all      = enc["attention_mask"]
    tti_all       = enc.get("token_type_ids", None)

    # create train/test/val datasets and dataloaders
    input_ids, attention_mask = {}, {}
    input_ids['train']       = input_ids_all[:nb_train]
    input_ids['val']         = input_ids_all[nb_train:nb_train+nb_val]
    input_ids['test']        = input_ids_all[-nb_test:] if nb_test > 0 else input_ids_all[:0]

    attention_mask['train']  = attn_all[:nb_train]
    attention_mask['val']    = attn_all[nb_train:nb_train+nb_val]
    attention_mask['test']   = attn_all[-nb_test:] if nb_test > 0 else attn_all[:0]

    # 4) keep dataset signature the same; stash token_type_ids on self
    if tti_all is not None:
        token_type_ids = {
            'train': tti_all[:nb_train],
            'val':   tti_all[nb_train:nb_train+nb_val],
            'test':  tti_all[-nb_test:] if nb_test > 0 else tti_all[:0],
        }
    else:
        token_type_ids = None

    datasets = {}
    loaders = {}

    for split in ['train', 'val', 'test']:
        datasets[split] = TensorDataset(input_ids[split], attention_mask[split], token_type_ids[split], label[split])
        loaders[split] = DataLoader(datasets[split], batch_size=batch_size, shuffle=True, worker_init_fn=self.seed)
    self.loaders = loaders

"""


"""
def load_custom_dataset_labels(config_data):
    #dataset_name = config_data.task_name
    dataset_name = config_data.dataset_name
    dataset_dir = f"/auto/k2/aykut4/kumbasar/1_bit_llm/baselines/custom_datasets"

    # Load shuffled doc names and cleaned text
    with open(f"{dataset_dir}/{dataset_name}_shuffle.txt") as f_meta, \
         open(f"{dataset_dir}/corpus/{dataset_name}_shuffle.txt") as f_text:
        meta_lines = [line.strip() for line in f_meta if line.strip()]
        text_lines = [line.strip() for line in f_text if line.strip()]

    # Load train/test indices
    with open(f"{dataset_dir}/{dataset_name}.train.index") as f_train, \
         open(f"{dataset_dir}/{dataset_name}.test.index") as f_test:
        train_indices = [int(line.strip()) for line in f_train if line.strip()]
        test_indices = [int(line.strip()) for line in f_test if line.strip()]

    print("meta_lines[:10]")
    print(meta_lines[:10])
    print("text_lines[:10]")
    print(text_lines[:10])
    print("train_indices[:10]")
    print(train_indices[:10])
    print("test_indices[:10]")
    print(test_indices[:10])

    # Extract labels from meta
    labels = []
    for meta in meta_lines:
        parts = meta.split('\t')
        labels.append(parts[2])  # third column is label

    # Load label list from file (ordered)
    #label_file = f"{dataset_dir}/corpus/{dataset_name}_labels.txt"
    #with open(label_file) as f_labels:
    #    label_list = [line.strip() for line in f_labels if line.strip()]
    label_list = [
        "C11", "C10", "C05", "C02", "C20", "C12", "C22", "C19", "C07", "C21", "C04",
        "C13", "C06", "C23", "C16", "C15", "C08", "C14", "C17", "C18", "C03", "C01", "C09"
    ]
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    print("label_list")
    print(label_list)
    print(f"Number of labels: {len(label_list)}")
    print("set(labels)")
    print(set(labels))
    print("label2id")
    print(label2id)

    # Build train/validation/test splits
    train_data = [{"text": text_lines[i], "label": label2id[labels[i]]} for i in train_indices]
    test_data = [{"text": text_lines[i], "label": label2id[labels[i]]} for i in test_indices]

    print("train_data[:10]")
    print(train_data[:10])
    print("test_data[:10]")
    print(test_data[:10])

    # Optionally, split train_data into train/validation (e.g., 90/10 split)
    split_idx = int(0.9 * len(train_data))
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(train_split)),
        "validation": Dataset.from_pandas(pd.DataFrame(val_split)),
        "test": Dataset.from_pandas(pd.DataFrame(test_data))
    })
    return raw_datasets, num_labels, label_list
"""
