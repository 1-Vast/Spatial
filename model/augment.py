from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def _as_csr_float32(X: sp.spmatrix, *, copy: bool = False) -> sp.csr_matrix:
    if sp.isspmatrix_csr(X):
        Y = X.copy() if copy else X
    else:
        Y = X.tocsr()
        if copy:
            Y = Y.copy()
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32)
    return Y

def row_l1_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    mat = _as_csr_float32(mat, copy=True)
    rowsum = np.asarray(mat.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0
    inv = 1.0 / rowsum
    mat.data *= np.repeat(inv, np.diff(mat.indptr))
    return mat

def topN_per_row(X: sp.csr_matrix, N: int) -> sp.csr_matrix:
    X = _as_csr_float32(X, copy=True)
    if N <= 0:
        return sp.csr_matrix(X.shape, dtype=np.float32)

    indptr = X.indptr
    data = X.data
    row_nnz = np.diff(indptr)
    if row_nnz.size == 0 or np.max(row_nnz) <= N:
        return X

    keep_mask = np.ones(data.shape[0], dtype=np.bool_)
    rows_to_trim = np.flatnonzero(row_nnz > N)

    for r in rows_to_trim:
        st = int(indptr[r])
        ed = int(indptr[r + 1])
        row_abs = np.abs(data[st:ed])
        kth = row_abs.size - N
        keep_rel = np.argpartition(row_abs, kth)[kth:]
        row_keep = np.zeros(row_abs.size, dtype=np.bool_)
        row_keep[keep_rel] = True
        keep_mask[st:ed] = row_keep

    keep_prefix = np.concatenate(([0], np.cumsum(keep_mask.astype(np.int64))))
    new_indptr = keep_prefix[indptr]
    X_top = sp.csr_matrix(
        (X.data[keep_mask], X.indices[keep_mask], new_indptr),
        shape=X.shape,
        dtype=np.float32,
    )
    X_top.sort_indices()
    X_top.eliminate_zeros()
    return X_top

def build_augmented_graph(
    A_s: sp.csr_matrix,
    X,
    alpha: float = 0.5,
    normalize_A: bool = True,
    normalize_X: bool = True,
    topN: int | None = None,
    *,
    spatial: np.ndarray | None = None,
    layer_aware: bool = False,
    layer_gamma: float = 3.0,
    layer_axis: int = 1,
    layer_scale_mode: str = "edge_median",
    layer_cutoff: float | None = None,
) -> sp.csr_matrix:

    assert A_s.shape[0] == (X.shape[0] if hasattr(X, "shape") else len(X))
    N = A_s.shape[0]

    A = _as_csr_float32(A_s, copy=False)

    if layer_aware:
        if spatial is None:
            raise ValueError("layer_aware=True but spatial is None")
        A = reweight_A_by_layer(
            A,
            spatial=spatial,
            gamma=layer_gamma,
            axis=layer_axis,
            scale_mode=layer_scale_mode,
            cutoff=layer_cutoff,
        )

    if normalize_A:
        A = row_l1_normalize(A + sp.eye(N, format="csr"))

    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    else:
        X = _as_csr_float32(X, copy=True)

    if topN is not None:
        X = topN_per_row(X, topN)

    if normalize_X:
        X = row_l1_normalize(X)

    G = X.shape[1]
    I = sp.eye(G, dtype=np.float32, format="csr")

    blocks = [
        [alpha * A,              (1 - alpha) * X           ],
        [(1 - alpha) * X.T,      alpha * I                 ],
    ]
    Pe = sp.bmat(blocks, format="csr")
    Pe.eliminate_zeros()
    return Pe

def reweight_A_by_layer(
    A: sp.csr_matrix,
    spatial: np.ndarray,
    gamma: float = 3.0,
    axis: int = 1,
    scale_mode: str = "edge_median",
    cutoff: float | None = None,
):

    A = _as_csr_float32(A, copy=False)
    row, col = A.nonzero()
    data = A.data.copy()

    dy = np.abs(spatial[row, axis] - spatial[col, axis]).astype(np.float32)

    if scale_mode == "global_std":
        scale = float(np.std(spatial[:, axis])) + 1e-6
    elif scale_mode == "edge_median":
        # local scale: typical dy of edges (more sensitive to layer boundary)
        scale = float(np.median(dy)) + 1e-6
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode}")

    # optional hard cutoff
    if cutoff is not None:
        thr = float(cutoff) * scale
        keep = dy <= thr
        row, col, data, dy = row[keep], col[keep], data[keep], dy[keep]

    decay = np.exp(-float(gamma) * dy / scale).astype(np.float32)
    new_data = (data * decay).astype(np.float32)

    A_new = sp.csr_matrix((new_data, (row, col)), shape=A.shape)
    A_new = 0.5 * (A_new + A_new.T)
    A_new.eliminate_zeros()
    return A_new
