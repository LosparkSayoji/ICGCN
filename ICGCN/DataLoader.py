import os
import numpy as np
import scipy.io as scio
import scipy.sparse as sp
import torch
from sklearn.neighbors import kneighbors_graph
from typing import Tuple, List, Dict


# --- Data Preprocessing Functions ---

def normalization(data: torch.Tensor) -> torch.Tensor:
    """
    Min-Max normalization.
    Note: Fixed the original // (integer division) to / (float division)
    to prevent zeroing out the data.
    """
    max_val = torch.max(data)
    min_val = torch.min(data)
    # Use float division to preserve features
    data = (data - min_val) / (max_val - min_val + 1e-10)
    return data


def standardization(data: torch.Tensor) -> torch.Tensor:
    """
    Row-wise L2 normalization.
    """
    row_sum = torch.sqrt(torch.sum(data ** 2, 1))
    rep_mat = row_sum.repeat((data.shape[1], 1)) + 1e-10
    data = torch.div(data, rep_mat.t())
    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    Uses coalesce() for improved compatibility.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # Using torch.sparse_coo_tensor (modern equivalent of FloatTensor)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def construct_laplacian(adj: sp.spmatrix) -> sp.coo_matrix:
    """
    Construct the Symmetric Normalized Laplacian matrix.
    L = I - D^-1/2 * A * D^-1/2
    """
    adj_ = sp.coo_matrix(adj)
    row_sum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())

    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = sp.eye(adj.shape[0]) - adj_wave
    return lp


def normalize(mx: sp.spmatrix) -> sp.coo_matrix:
    """
    Row-normalize sparse matrix: D^-1 * A
    """
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# --- Data Loading ---

def load_mat_data(datasets: str, k: int, path: str = "./data/"):
    """
    Load multi-view .mat data and strictly follow original KNN construction.
    """
    print(f"Loading {datasets} data...")
    data = scio.loadmat(f"{path}{datasets}.mat")
    x = data["X"]

    adj_list = []
    features_list = []
    nfeats_list = []

    num_views = x.shape[1]
    for i in range(num_views):
        view_features = x[0, i]
        if datasets == '100leaves':
            view_features = np.transpose(view_features)

        view_features = view_features.astype('float32')
        features_list.append(view_features)
        nfeats_list.append(view_features.shape[1])

        # Strictly maintain original KNN symmetrization logic
        temp = kneighbors_graph(view_features, k)
        temp = sp.coo_matrix(temp)
        # Original logic: A + A.T * (A.T > A) - A * (A.T > A)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)

        # Normalize with self-loops
        temp = normalize(temp + sp.eye(temp.shape[0]))
        temp = sparse_mx_to_torch_sparse_tensor(temp)
        adj_list.append(temp)

    # Strictly maintain original label subtraction logic
    labels = data["Y"].reshape(-1, ).astype('int64')
    labels = labels - min(set(labels))
    num_class = len(set(np.array(labels)))

    # Cast to .long() to prevent CUDA RuntimeError for indexing
    return adj_list, features_list, torch.from_numpy(labels).long(), nfeats_list, num_views, num_class


# --- Split and Utilities ---

def count_each_class_num(gnd: np.ndarray) -> Dict:
    """
    Count the number of samples per class.
    """
    unique, counts = np.unique(gnd, return_counts=True)
    return dict(zip(unique, counts))


def generate_splits(gnd: torch.Tensor, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Strictly follows original ratio-based or fixed-number split logic.
    """
    gnd_np = np.array(gnd)
    N = gnd_np.shape[0]
    each_class_num = count_each_class_num(gnd_np)

    training_each_class_num = {}
    test_num = 0

    # Determine samples per class
    for label in each_class_num.keys():
        if hasattr(args, 'data_split_mode') and args.data_split_mode == "Ratio":
            train_ratio = args.train_ratio
            test_ratio = 1 - args.train_ratio
            training_each_class_num[label] = max(round(each_class_num[label] * train_ratio), 1)
            test_num = max(round(N * test_ratio), 1)
        else:
            training_each_class_num[label] = args.num_train_per_class
            test_num = args.num_test

    train_mask = np.full(N, False)
    train_idx = []
    test_idx = []

    # Random permutation
    data_idx = np.random.permutation(range(N))

    # Assign training samples
    for idx in data_idx:
        label = gnd_np[idx]
        if training_each_class_num[label] > 0:
            training_each_class_num[label] -= 1
            train_mask[idx] = True
            train_idx.append(idx)

    # Assign test samples from remaining pool
    for idx in data_idx:
        if train_mask[idx]:
            continue
        if test_num > 0:
            test_num -= 1
            test_idx.append(idx)

    return torch.tensor(train_idx).long(), torch.tensor(test_idx).long()
