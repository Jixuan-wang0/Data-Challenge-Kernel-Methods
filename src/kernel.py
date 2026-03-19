"""
Kernel.py - RBF kernel construction.
"""

import numpy as np


def _sq_dists(A, B):
    """
    Compute squared Euclidean distances between rows of A (N,D) and B (M,D).
    """
    AA = np.einsum('ij,ij->i', A, A)[:, None]   # (N,1)
    BB = np.einsum('ij,ij->i', B, B)[None, :]   # (1,M)
    AB = A @ B.T                                # (N,M)
    D2 = AA + BB - 2.0 * AB
    np.clip(D2, 0.0, None, out=D2)               # numerical safety (can get small negatives)
    return D2


def rbf_kernel(A, B, gamma):
    """
    RBF kernel: exp(-gamma * ||x - y||^2)
    """
    return np.exp(-gamma * _sq_dists(A, B)).astype(np.float64)


def normalise_gram(K_train, K_rect=None):
    """
    Normalize kernel matrix.
    """
    d_train = np.sqrt(np.maximum(np.diag(K_train), 1e-12))

    if K_rect is None:
        return K_train / np.outer(d_train, d_train)

    return K_rect / d_train[np.newaxis, :]


def build_gram_matrices(Xtr, Xte, gamma):
    """
    Build and return normalised Gram matrices.
    """
    print(f"Building K_train ({Xtr.shape[0]}x{Xtr.shape[0]})...")
    K_train_raw = rbf_kernel(Xtr, Xtr, gamma)
    K_train     = normalise_gram(K_train_raw)

    print(f"Building K_test  ({Xte.shape[0]}x{Xtr.shape[0]})...")
    K_test_raw  = rbf_kernel(Xte, Xtr, gamma)
    K_test      = normalise_gram(K_train_raw, K_test_raw)

    return K_train, K_test
