import numpy as np


def _connectivity(V, idx):
    idx = np.asmatrix(idx)
    mat1 = np.tile(idx, (V.shape[1], 1))
    mat2 = np.tile(idx.T, (1, V.shape[1]))
    conn = np.equal(np.mat(mat1), np.mat(mat2))
    return np.mat(conn, dtype='d')


def connectivity(V, H):  # derived from nimfa
    idx = np.asmatrix(np.argmax(H, axis=0))
    mat1 = np.tile(idx, (V.shape[1], 1))
    mat2 = np.tile(idx.T, (1, V.shape[1]))
    conn = np.equal(np.mat(mat1), np.mat(mat2))
    return np.mat(conn, dtype='d')
