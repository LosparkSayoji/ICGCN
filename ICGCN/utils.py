import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn import manifold
from typing import Union, Optional, Callable


# --- Metrics ---

def calculate_accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the accuracy for a given set of predictions and labels.
    """
    preds = output.argmax(dim=1).type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)


def calculate_f1(output: torch.Tensor, labels: torch.Tensor, average: str = 'macro') -> float:
    """
    Computes the F1 score using sklearn.
    """
    preds = output.argmax(dim=1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()
    return f1_score(true_labels, preds, average=average)


# --- Sparse Matrix Transformations ---

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.coo_matrix, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Converts a Scipy sparse matrix to a Torch sparse COO tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    tensor = torch.sparse_coo_tensor(indices, values, shape)
    if device:
        tensor = tensor.to(device)
    return tensor.coalesce()


def row_normalize(mx: sp.spmatrix) -> sp.coo_matrix:
    """
    Performs row-normalization: D^-1 * A
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx).tocoo()


def aug_normalized_adjacency(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.coo_matrix:
    """
    Computes the symmetrically normalized adjacency matrix:
    A' = (D + I)^-1/2 * (A + I) * (D + I)^-1/2
    """
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0])

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


# --- Visualization ---

def plot_embeddings(output: torch.Tensor, labels: torch.Tensor, title: str = "T-SNE Visualization"):
    """
    Visualizes node embeddings using T-SNE.
    """
    # Prepare data
    embeddings = output.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Run T-SNE
    tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)
    x_tsne = tsne.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels_np, cmap='tab10', s=15, alpha=0.8)

    # Dynamic Legend
    unique_labels = np.unique(labels_np)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f'Class {int(i)}' for i in unique_labels], loc="best", title="Labels")

    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# --- Factory Pattern for Normalization ---

def get_normalization_method(method_name: str) -> Callable:
    """
    Returns the appropriate normalization function based on string input.
    """
    methods = {
        'AugNormAdj': aug_normalized_adjacency,
        'RowNorm': row_normalize
    }

    if method_name not in methods:
        raise ValueError(f"Normalization method '{method_name}' not recognized. Available: {list(methods.keys())}")

    return methods[method_name]