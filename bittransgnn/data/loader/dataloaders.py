import random

import torch
from torch.utils.data import DataLoader, TensorDataset

import pickle as pkl

import numpy as np
import scipy.sparse as sp

from ..preprocessing.data_preprocessing import normalize_adj, parse_index_file, sample_mask

SEP_WORD = "<<SEP>>"
PAIR_DATASETS = ["mrpc", "rte", "stsb"]
BASE = "dataset"
BASE_PAIR = "dataset_paired"

def split_ab(line: str, sep: str = SEP_WORD):
    line = line.strip()
    if not line:
        return "", None
    if sep in line:
        a, b = line.split(sep, 1)
        return a.strip(), b.strip()
    return line, None  # single-sentence datasets

def load_corpus(dataset_name, adj_type="full"):
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
    if dataset_name in PAIR_DATASETS:
        base_path = BASE_PAIR
    else:
        base_path = BASE
    adj_name = "adj"
    if adj_type != "full":
        adj_name += f"_{adj_type}"
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', adj_name]
    objects = []

    for i in range(len(names)):
        with open(f"{base_path}/ind.{dataset_name}.{names[i]}", 'rb') as f:
            objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
    print(adj.shape)

    train_idx_orig = parse_index_file(
        f"{base_path}/{dataset_name}.train.index")
    train_size = len(train_idx_orig)
    print("train_size: ", train_size)

    if adj_type != "doc_doc":
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))
    else:
        # Combine all document features and labels (exclude word nodes)
        features = sp.vstack((allx[:train_size], tx)).tolil()  # Only include document nodes
        labels = np.vstack((ally[:train_size], ty))  # Only include document labels
        print("Document-only features shape:", features.shape)
        print("Document-only labels shape:", labels.shape)

    print(len(labels))

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]
    print("val_size: ", val_size)
    print("test_size: ", test_size)

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    if adj_type != "doc_doc":
        idx_test = range(allx.shape[0], allx.shape[0] + test_size)
    else:
        idx_test = range(len(y)+val_size, len(y) + val_size + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

def load_train_corpus(dataset_name, adj_type="full"):
    """
    Loads input corpus from ./dataset directory based on dataset_name. Loads the training set only for inductive learning setting.

    ind.dataset_name.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_name.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_name.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_name.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_name.ally => the labels for instances in ind.dataset_name.allx as numpy.ndarray object;
    ind.dataset_name.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_name.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_name: Dataset name
    :param adj_type: None (full), no-doc (ww+wd), no-ww(dd+wd), doc-doc (only dd)
    :return: All data input files loaded.
    """
    if dataset_name in PAIR_DATASETS:
        base_path = BASE_PAIR
    else:
        base_path = BASE
    adj_name = "adj"
    if adj_type != "full":
        adj_name += f"_{adj_type}"
    names = ['x', 'y', 'allx', 'ally', adj_name]
    objects = []
    for i in range(len(names)):
        with open(f"{base_path}/ind.{dataset_name}_inductive.{names[i]}", 'rb') as f:
            objects.append(pkl.load(f))

    x, y, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, allx.shape, ally.shape)

    features = allx.tolil()
    labels = ally
    print(len(labels))

    train_idx_orig = parse_index_file(
        f"{base_path}/ind.{dataset_name}_inductive.train.index")
    train_size = len(train_idx_orig)
    print("train_size: ", train_size)

    val_size = train_size - x.shape[0]
    print("val_size: ", val_size)

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, train_mask, val_mask, train_size

class TextDataObject:
    def __init__(self, dataset_name, batch_size, adj_type="full", seed=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.adj_type = adj_type
        self.seed = seed
        self.nb_class, self.nb_train, self.nb_val, self.nb_test, self.label = self.load_labels()

    def load_labels(self):
        """
        Returns the dataloaders for BERT-based models
        """
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(self.dataset_name, self.adj_type)
        # compute number of real train/val/test/word nodes and number of classes
        nb_node = adj.shape[0]
        nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        nb_word = nb_node - nb_train - nb_val - nb_test
        nb_class = y_train.shape[1]
        # transform one-hot label to class ID for pytorch computation
        if self.dataset_name == "stsb":
            y = y_train + y_val + y_test
            y = torch.from_numpy(y).to(dtype=torch.float32)
        else:
            y = torch.LongTensor((y_train + y_val +y_test).argmax(axis=1))
        label = {}
        label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]
        return nb_class, nb_train, nb_val, nb_test, label

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

class GraphDataObject:
    def __init__(self, dataset_name, batch_size, adj_type="full", seed=None, train_only=False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.adj_type = adj_type
        self.train_only = train_only
        self.seed = seed
        if train_only:
            self.idx_loaders, self.adj, self.y_train, self.y_val, self.train_mask, self.val_mask, self.nb_class = self.get_dataloaders_train_only_bertgcn()
        else:
            self.idx_loaders, self.adj, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask, self.nb_class = self.get_dataloaders_bertgcn()
        self.label, self.train_label = self.get_output_labels()
        if train_only:
            self.doc_mask, self.train_mask, self.val_mask = self.get_masks()
        else:
            self.doc_mask, self.train_mask, self.val_mask, self.test_mask = self.get_masks()

    def get_dataloaders_bertgcn(self):
        """
        Returns the dataloaders for BERTGCN-based models
        """
        dataset_name, batch_size = self.dataset_name, self.batch_size
        # Data Preprocess
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(self.dataset_name, self.adj_type)
        '''
        adj: n*n sparse adjacency matrix
        y_train, y_val, y_test: n*c matrices 
        train_mask, val_mask, test_mask: n-d bool array
        '''

        # compute number of real train/val/test/word nodes and number of classes
        #nb_node = features.shape[0]
        nb_node = adj.shape[0]
        nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        nb_word = nb_node - nb_train - nb_val - nb_test
        nb_class = y_train.shape[1]

        print("nb_node: ", nb_node)
        print("nb_train: ", nb_train)
        print("nb_val: ", nb_val)
        print("nb_word: ", nb_word)

        # create index loader
        train_idx = TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
        val_idx = TensorDataset(torch.arange(nb_train, nb_train + nb_val, dtype=torch.long))
        test_idx = TensorDataset(torch.arange(nb_node-nb_test, nb_node, dtype=torch.long))

        idx_loader_train = DataLoader(train_idx, batch_size=batch_size, shuffle=True, worker_init_fn=self.seed)
        idx_loader_val = DataLoader(val_idx, batch_size=batch_size, worker_init_fn=self.seed)
        idx_loader_test = DataLoader(test_idx, batch_size=batch_size, worker_init_fn=self.seed)
        
        idx_loaders = {}
        idx_loaders["train"], idx_loaders["val"], idx_loaders["test"] = idx_loader_train, idx_loader_val, idx_loader_test
        #idx_loader = DataLoader(doc_idx, batch_size=batch_size, shuffle=True)
        return idx_loaders, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask, nb_class

    def get_dataloaders_train_only_bertgcn(self):
        """
        Returns the dataloaders for BERTGCN-based models when only the training set is used for graph construction
        """
        dataset_name, batch_size, adj_type = self.dataset_name, self.batch_size, self.adj_type
        # Data Preprocess
        adj, features, y_train, y_val, train_mask, val_mask, train_size = load_train_corpus(dataset_name, adj_type)
        dataset_name += "_inductive"
        '''
        adj: n*n sparse adjacency matrix
        y_train, y_val: n*c matrices 
        train_mask, val_mask: n-d bool array
        '''

        # compute number of real train/val/test/word nodes and number of classes
        nb_node = adj.shape[0]
        nb_train, nb_val = train_mask.sum(), val_mask.sum()
        nb_word = nb_node - nb_train - nb_val
        nb_class = y_train.shape[1]
        
        # create index loader
        train_idx = TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
        val_idx = TensorDataset(torch.arange(nb_train, nb_train + nb_val, dtype=torch.long))

        idx_loader_train = DataLoader(train_idx, batch_size=batch_size, shuffle=True, worker_init_fn=self.seed)
        idx_loader_val = DataLoader(val_idx, batch_size=batch_size, worker_init_fn=self.seed)
        
        idx_loaders = {}
        idx_loaders["train"], idx_loaders["val"] = idx_loader_train, idx_loader_val
        #idx_loader = DataLoader(doc_idx, batch_size=batch_size, shuffle=True)
        return idx_loaders, adj, y_train, y_val, train_mask, val_mask, nb_class

    def get_output_labels(self):
        y_train, y_test, y_val = self.y_train, self.y_test, self.y_val
        # transform one-hot labels to class IDs to make it usable during pytorch computations
        if self.dataset_name == "stsb":
            if self.train_only:
                y = y_train + y_val
            else:
                y = y_train + y_val + y_test
            y = torch.from_numpy(y).to(dtype=torch.float32)
            y_train = torch.from_numpy(y_train).to(dtype=torch.float32)
        else:
            if self.train_only:
                y = y_train + y_val
            else:
                y = y_train + y_test + y_val
            y_train = y_train.argmax(axis=1)
            y = y.argmax(axis=1)
            y = torch.LongTensor(y)
            y_train = torch.LongTensor(y_train)
        return y, y_train
    
    def get_masks(self):
        train_mask, val_mask, test_mask = self.train_mask, self.val_mask, self.test_mask
        # document mask used to update features
        doc_mask = train_mask + val_mask
        if not(self.train_only):
            doc_mask += test_mask
            test_mask = torch.FloatTensor(test_mask)
        train_mask = torch.FloatTensor(train_mask)
        val_mask = torch.FloatTensor(val_mask)
        doc_mask = torch.FloatTensor(doc_mask)
        if self.train_only:
            return doc_mask, train_mask, val_mask
        else:
            return doc_mask, train_mask, val_mask, test_mask
    
    def set_transformer_data(self, model, max_length):
        train_mask, val_mask = self.train_mask, self.val_mask
        if not(self.train_only):
            test_mask = self.test_mask
        dataset_name = self.dataset_name
        adj = self.adj
        nb_node = adj.shape[0]
        nb_train, nb_val = train_mask.sum(), val_mask.sum()
        if self.train_only:
            nb_word = nb_node - nb_train - nb_val
        else:
            nb_test = test_mask.sum()
            nb_word = nb_node - nb_train - nb_val - nb_test
            nb_word, nb_test = int(nb_word), int(nb_test)
        # load documents and compute input encodings for BERT
        if self.train_only:
            dataset_name += "_inductive"
        if dataset_name in PAIR_DATASETS:
            base_path = BASE_PAIR
        else:
            base_path = BASE
        corpse_file = f'{base_path}/corpus/{dataset_name}_shuffle.txt'
        with open(corpse_file, 'r') as f:
            text = f.read().replace('\\', '').split('\n')

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
            
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        tti            = enc.get("token_type_ids", None)

        if self.adj_type != "doc_doc":
            if self.train_only:
                input_ids = torch.cat([input_ids, torch.zeros((nb_word, max_length), dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros((nb_word, max_length), dtype=torch.long)])
                if tti is not None:
                    tti = torch.cat([tti, torch.zeros((nb_word, max_length), dtype=torch.long)])
            else:
                input_ids = torch.cat([input_ids[:-nb_test], torch.zeros((nb_word, max_length), dtype=torch.long), input_ids[-nb_test:]])
                attention_mask = torch.cat([attention_mask[:-nb_test], torch.zeros((nb_word, max_length), dtype=torch.long), attention_mask[-nb_test:]])
                if tti is not None:
                    tti = torch.cat([tti[:-nb_test], torch.zeros((nb_word, max_length), dtype=torch.long), tti[-nb_test:]])
        else:
            input_ids = input_ids.to(dtype=torch.long)
            attention_mask = attention_mask.to(dtype=torch.long)
            if tti is not None:
                tti = tti.to(dtype=torch.long)

        self.input_ids, self.attention_mask = input_ids, attention_mask
        self.token_type_ids = tti  # optional: available if you want it in forward()

    def set_graph_data(self, model):
        adj = self.adj
        nb_node = adj.shape[0]
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        cls_feats = torch.zeros((nb_node, model.feat_dim))
        adj_coo = adj_norm.tocoo()
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))
        idxs = torch.LongTensor(indices)
        vals = torch.FloatTensor(values)
        coo_shape = adj_coo.shape
        adj_sparse = torch.sparse_coo_tensor(idxs, vals, size=coo_shape)
        self.cls_feats, self.adj_sparse = cls_feats, adj_sparse

    def convert_device(self, device):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device)) 
        return self
