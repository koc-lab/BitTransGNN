import numpy as np
import scipy.sparse as sp

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """
    Preprocessing of adjacency matrix for simple GCN model.
    Self-loops are added to the original adjacency as a preprocessing step.
                   \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}
    Then, the adjacency matrix is normalized through symmetric normalization, as:
       \mathbf{A} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

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
