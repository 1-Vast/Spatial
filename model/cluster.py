import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter

import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

# Optional rpy2 for R::Mclust
try:
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri as numpy2ri

    # numpy2ri.activate()
    HAS_RPY2 = True
except Exception:
    HAS_RPY2 = False


# ----------------------------------------------------------------------
# Basic graph utilities
# ----------------------------------------------------------------------
def _ensure_undirected(A: sp.spmatrix) -> sp.spmatrix:
    A = A.tocsr()
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _row_norm(A: sp.spmatrix) -> sp.csr_matrix:
    A = A.tocsr().astype(np.float32)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv = sp.diags(1.0 / deg)
    return inv @ A


def _neighbors_from_matrix_gpu(
    X: np.ndarray,
    n_neighbors: int,
    gpu_device: str = "cuda",
) -> sp.csr_matrix:
    try:
        import torch
        from torch_geometric.nn import knn_graph
    except Exception as e:
        raise RuntimeError("GPU neighbors backend requires torch + torch_geometric.") from e

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in current environment.")

    X = np.asarray(X, dtype=np.float32)
    n_obs = int(X.shape[0])
    if n_obs <= 1:
        return csr_matrix((n_obs, n_obs), dtype=np.float32)

    k_eff = max(1, min(int(n_neighbors), n_obs - 1))
    x_t = torch.tensor(X, dtype=torch.float32, device=gpu_device)
    edge_index = knn_graph(x_t, k=k_eff, loop=False, flow="target_to_source")
    edge_index = edge_index.detach().cpu().numpy()
    row, col = edge_index[0], edge_index[1]
    data = np.ones(row.shape[0], dtype=np.float32)
    A = csr_matrix((data, (row, col)), shape=(n_obs, n_obs), dtype=np.float32)
    A = _ensure_undirected(A)
    A = A.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _set_neighbors_graph(adata, A: sp.spmatrix, backend: str, n_neighbors: int):
    A = _ensure_undirected(A).tocsr().astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    adata.obsp["connectivities"] = A
    adata.obsp["distances"] = A.copy()
    adata.uns["neighbors"] = {
        "params": {"method": str(backend), "n_neighbors": int(n_neighbors)},
        "connectivities_key": "connectivities",
        "distances_key": "distances",
    }


def _neighbors_from_rep(
    adata,
    use_rep: str,
    n_neighbors: int,
    backend: str = "scanpy",
    gpu_device: str = "cuda",
):
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm['{use_rep}'] not found.")

    backend = str(backend).lower().strip()
    if backend in ("gpu", "auto"):
        try:
            A = _neighbors_from_matrix_gpu(adata.obsm[use_rep], n_neighbors, gpu_device=gpu_device)
            _set_neighbors_graph(adata, A, backend=f"gpu:{gpu_device}", n_neighbors=n_neighbors)
            return
        except Exception as e:
            if backend == "gpu":
                raise
            print(f"[neighbors] GPU backend unavailable ({e}); fallback to scanpy.", flush=True)

    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=int(n_neighbors))


def _neighbors_from_embedding(
    adata,
    Z: np.ndarray,
    knn_k: int = 15,
    backend: str = "scanpy",
    gpu_device: str = "cuda",
):
    """
    Build neighbors based on embedding Z and store in adata.obsp["connectivities"].
    """
    backend = str(backend).lower().strip()
    if backend in ("gpu", "auto"):
        try:
            A = _neighbors_from_matrix_gpu(Z, knn_k, gpu_device=gpu_device)
            _set_neighbors_graph(adata, A, backend=f"gpu:{gpu_device}", n_neighbors=knn_k)
            return
        except Exception as e:
            if backend == "gpu":
                raise
            print(f"[neighbors] GPU backend unavailable ({e}); fallback to scanpy.", flush=True)

    adata.obsm["X_tmp"] = Z
    sc.pp.neighbors(adata, use_rep="X_tmp", n_neighbors=int(knn_k))
    del adata.obsm["X_tmp"]


def build_hybrid_adj(
    adata,
    n_neighbors_emb: int = 12,
    use_rep: str = "X",
    w_spa: float = 0.7,
    w_emb: float = 0.3,
    neighbors_backend: str = "scanpy",
    neighbors_gpu_device: str = "cuda",
):
    assert "A_spatial" in adata.obsp, "missing adata.obsp['A_spatial']"
    A_spa = _ensure_undirected(adata.obsp["A_spatial"])

    cache_backend = str(neighbors_backend).lower().strip()
    rep_mat = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    rep_shape = tuple(rep_mat.shape)
    rep_n = int(rep_shape[0]) if len(rep_shape) > 0 else int(adata.n_obs)
    rep_d = int(rep_shape[1]) if len(rep_shape) > 1 else 1
    cache_key = (
        f"A_emb_{str(use_rep)}_k{int(n_neighbors_emb)}_{cache_backend}"
        f"_n{rep_n}_d{rep_d}"
    )
    if cache_key in adata.obsp:
        A_emb = _ensure_undirected(adata.obsp[cache_key])
    else:
        _neighbors_from_rep(
            adata,
            use_rep=use_rep,
            n_neighbors=n_neighbors_emb,
            backend=neighbors_backend,
            gpu_device=neighbors_gpu_device,
        )
        A_emb = _ensure_undirected(adata.obsp["connectivities"])
        adata.obsp[cache_key] = A_emb.copy()

    A_hyb = _row_norm(A_spa).multiply(w_spa) + _row_norm(A_emb).multiply(w_emb)
    A_hyb = A_hyb.tocsr()
    A_hyb.setdiag(0)
    A_hyb.eliminate_zeros()
    return A_spa, A_hyb


# ----------------------------------------------------------------------
# Power embedding smoothing (MAEST-style multi-hop propagation)
# ----------------------------------------------------------------------
def power_smooth_embedding(
    Z: np.ndarray,
    A_spa: sp.spmatrix,
    power: int = 0,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Multi-hop feature propagation on spatial graph A_spa.

    local = original embedding
    H = A^power @ local      (A is row-normalized)
    final = local + alpha * H
    """
    Z = np.asarray(Z, dtype=np.float32)
    if power <= 0 or A_spa is None:
        return Z

    A = _row_norm(_ensure_undirected(A_spa))
    H_local = Z.copy()
    H = Z.copy()

    for _ in range(int(power)):
        H = A @ H

    H_final = H_local + float(alpha) * H
    return H_final.astype(np.float32)


# ----------------------------------------------------------------------
# Simple spatial KNN smoothing for embeddings (BEFORE clustering)
# ----------------------------------------------------------------------
def spatial_knn_smooth_embedding(
    Z: np.ndarray,
    spatial: np.ndarray,
    k: int = 20,
    n_iter: int = 1,
) -> np.ndarray:
    """
    Spatial KNN feature smoothing (MAEST-like preprocessing):
      Z_new[i] = mean( Z[neighbors(i) ∪ {i}] )

    Notes:
      - Uses spatial coordinates KNN (not expression/embedding KNN).
      - Applies n_iter times.
    """
    Z = np.asarray(Z, dtype=np.float32)
    if spatial is None:
        raise ValueError("spatial is None, cannot do spatial KNN smoothing.")
    if spatial.shape[0] != Z.shape[0]:
        raise ValueError(f"spatial.shape[0]={spatial.shape[0]} != Z.shape[0]={Z.shape[0]}")

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError("sklearn is required for spatial KNN smoothing.") from e

    k = int(k)
    n_iter = int(n_iter)
    if k <= 0 or n_iter <= 0:
        return Z

    nn = NearestNeighbors(n_neighbors=min(k + 1, Z.shape[0]), algorithm="auto")
    nn.fit(spatial[:, :2])
    neigh_ind = nn.kneighbors(return_distance=False)

    Z_s = Z.copy()
    for _ in range(n_iter):
        Z_new = np.empty_like(Z_s)
        for i in range(Z_s.shape[0]):
            idx = neigh_ind[i]
            Z_new[i] = Z_s[idx].mean(axis=0)
        Z_s = Z_new
    return Z_s.astype(np.float32)


def mrf_majority_smooth(A_spa: sp.spmatrix, y: np.ndarray, n_iter: int = 2) -> np.ndarray:
    A = _ensure_undirected(A_spa).tocsr()
    y = np.asarray(y).copy()
    n_iter = int(n_iter)
    if n_iter <= 0 or y.size == 0:
        return y

    # Keep the original "neighbor count vote" semantics (ignore edge weights).
    A_bin = A.copy().tocsr()
    A_bin.data = np.ones_like(A_bin.data, dtype=np.float32)
    deg = np.diff(A_bin.indptr).astype(np.float32)

    uniq_labels, y_idx = np.unique(y, return_inverse=True)
    n_nodes = y_idx.shape[0]
    n_classes = uniq_labels.shape[0]

    for _ in range(n_iter):
        onehot = np.zeros((n_nodes, n_classes), dtype=np.float32)
        onehot[np.arange(n_nodes), y_idx] = 1.0
        votes = np.asarray(A_bin @ onehot)

        major_idx = votes.argmax(axis=1)
        major_cnt = votes[np.arange(n_nodes), major_idx]
        support = major_cnt >= np.maximum(2.0, 0.5 * deg)

        y_new_idx = y_idx.copy()
        y_new_idx[support] = major_idx[support]
        y_idx = y_new_idx

    return uniq_labels[y_idx]


# ----------------------------------------------------------------------
# Spatial refinement for labels using spatial KNN (refine_k)
# ----------------------------------------------------------------------
def spatial_knn_refine_labels(
    labels: np.ndarray,
    spatial: np.ndarray,
    k: int = 50,
    n_iter: int = 1,
) -> np.ndarray:
    """
    Majority-vote refinement using spatial KNN graph (by coordinates).
    Useful when you don't want to rely on A_spatial, or you want a controlled k.
    """
    labels = np.asarray(labels, dtype=int).copy()
    if spatial is None:
        raise ValueError("spatial is None, cannot do spatial KNN label refinement.")
    if spatial.shape[0] != labels.shape[0]:
        raise ValueError(f"spatial.shape[0]={spatial.shape[0]} != labels.shape[0]={labels.shape[0]}")

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError("sklearn is required for spatial KNN refinement.") from e

    k = int(k)
    n_iter = int(n_iter)
    if k <= 0 or n_iter <= 0:
        return labels

    nn = NearestNeighbors(n_neighbors=min(k + 1, labels.shape[0]), algorithm="auto")
    nn.fit(spatial[:, :2])
    neigh_ind = nn.kneighbors(return_distance=False)

    y = labels
    for _ in range(n_iter):
        y_new = y.copy()
        for i in range(y.shape[0]):
            vs = y[neigh_ind[i]]
            major, _cnt = Counter(vs.tolist()).most_common(1)[0]
            y_new[i] = int(major)
        y = y_new
    return y.astype(int)


# ----------------------------------------------------------------------
# Robust consensus Leiden on hybrid adjacency
# ----------------------------------------------------------------------
def _estimate_k(n: int) -> int:
    if n <= 20:
        return 2
    if n <= 100:
        return 3
    if n <= 500:
        return 5
    if n <= 2000:
        return 7
    if n <= 10000:
        return 8
    return 10


def _resolve_scale_mode(n_obs: int, scale_mode: str) -> str:
    mode = str(scale_mode).lower().strip()
    if mode in ("small", "medium", "large"):
        return mode
    if n_obs <= 10000:
        return "small"
    if n_obs <= 50000:
        return "medium"
    return "large"


def _has_cli_flag(flag: str) -> bool:
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(flag + "="):
            return True
    return False


def _run_leiden(adata, **kwargs):
    sc.tl.leiden(
        adata,
        flavor="igraph",
        directed=False,
        n_iterations=2,
        **kwargs,
    )


def run_leiden_on_adjacency(adata, adjacency: sp.spmatrix, resolution: float, key: str, seed: int):
    # 1) First try: newer scanpy supports adjacency=
    try:
        _run_leiden(
            adata,
            resolution=float(resolution),
            key_added=key,
            adjacency=adjacency,
            random_state=int(seed),
        )
        adata.obs[key] = adata.obs[key].astype(int)
        return
    except TypeError:
        # 2) Fallback: older scanpy doesn't accept adjacency=
        pass

    # --- backup existing neighbor graph (if any) ---
    old_conn = adata.obsp.get("connectivities", None)
    old_dist = adata.obsp.get("distances", None)
    old_uns_neighbors = adata.uns.get("neighbors", None)

    try:
        # Put custom adjacency into the standard slot scanpy expects
        adata.obsp["connectivities"] = adjacency.tocsr()

        # distances isn't strictly required for leiden, but some utilities expect it
        # We provide a placeholder sparse matrix with same sparsity/shape
        adata.obsp["distances"] = adata.obsp["connectivities"].copy()

        # Minimal neighbors metadata so scanpy knows where to look
        adata.uns["neighbors"] = {
            "params": {"method": "custom", "random_state": int(seed)},
            "connectivities_key": "connectivities",
            "distances_key": "distances",
        }

        _run_leiden(
            adata,
            resolution=float(resolution),
            key_added=key,
            random_state=int(seed),
        )
        adata.obs[key] = adata.obs[key].astype(int)
    finally:
        # --- restore ---
        if old_conn is None:
            if "connectivities" in adata.obsp: del adata.obsp["connectivities"]
        else:
            adata.obsp["connectivities"] = old_conn

        if old_dist is None:
            if "distances" in adata.obsp: del adata.obsp["distances"]
        else:
            adata.obsp["distances"] = old_dist

        if old_uns_neighbors is None:
            if "neighbors" in adata.uns: del adata.uns["neighbors"]
        else:
            adata.uns["neighbors"] = old_uns_neighbors


def robust_consensus_leiden(
    adata,
    use_rep: str,
    k_emb: int,
    key_added: str,
    w_spa: float = 0.9,
    w_emb: float = 0.2,
    base_resolution: float | None = None,
    res_list=None,
    n_clusters: int | None = None,
    n_seeds: int = 5,
    n_smooth_iter: int = 2,
    random_state: int = 0,
    consensus_backend: str = "kmeans",
    consensus_batch_size: int = 8192,
    neighbors_backend: str = "scanpy",
    neighbors_gpu_device: str = "cuda",
    scale_mode: str = "auto",
):
    """
    Robust consensus Leiden on hybrid adjacency:
    - For each seed and resolution (if res_list is not None) or single resolution:
      run Leiden on A_hyb
    - Collect all labels into one-hot and cluster in this "meta" space (KMeans)
    """
    if res_list is None:
        if base_resolution is None:
            base_resolution = 0.6
        res_list = [base_resolution]

    A_spa, A_hyb = build_hybrid_adj(
        adata,
        n_neighbors_emb=int(k_emb),
        use_rep=use_rep,
        w_spa=w_spa,
        w_emb=w_emb,
        neighbors_backend=neighbors_backend,
        neighbors_gpu_device=neighbors_gpu_device,
    )

    labels_all = []
    N = adata.n_obs
    key_prefix = f"{key_added}_tmp"

    seeds = [random_state + i for i in range(n_seeds)]
    for seed in seeds:
        for res in res_list:
            key_run = f"{key_prefix}_s{seed}_r{res:.3f}"
            run_leiden_on_adjacency(
                adata,
                adjacency=A_hyb,
                resolution=float(res),
                key=key_run,
                seed=int(seed),
            )
            labels_all.append(adata.obs[key_run].to_numpy().astype(int))
            del adata.obs[key_run]

    if not labels_all:
        raise RuntimeError("No Leiden runs were performed.")
    labels_all = np.stack(labels_all, axis=0)  # [R, N]

    run_dims = []
    for lab in labels_all:
        lab = lab.astype(int)
        run_dims.append(int(lab.max()) + 1)

    total_dim = int(np.sum(run_dims))
    meta_feat = np.zeros((N, total_dim), dtype=np.float32)
    offset = 0
    for lab, dim_i in zip(labels_all, run_dims):
        meta_feat[np.arange(N), offset + lab] = 1.0
        offset += dim_i

    if n_clusters is not None and n_clusters > 0:
        k_final = int(n_clusters)
    else:
        k_list = [len(np.unique(lab)) for lab in labels_all]
        k_final = int(np.median(k_list))
        k_final = max(2, k_final)
        print(f"[robust] k_list={k_list} -> k_final={k_final}", flush=True)

    backend = str(consensus_backend).lower().strip()
    if backend == "minibatch":
        km = MiniBatchKMeans(
            n_clusters=k_final,
            random_state=random_state,
            batch_size=max(256, int(consensus_batch_size)),
            n_init=3,
        )
    else:
        km = KMeans(n_clusters=k_final, n_init=10, random_state=random_state)
    y_robust = km.fit_predict(meta_feat).astype(int)

    y_robust_smooth = mrf_majority_smooth(A_spa, y_robust, n_iter=n_smooth_iter)

    adata.obs[key_added] = y_robust_smooth
    adata.obs[key_added] = adata.obs[key_added].astype(int)
    adata.uns[f"{key_added}_params"] = {
        "method": "robust_consensus_leiden",
        "k_emb": int(k_emb),
        "w_spa": float(w_spa),
        "w_emb": float(w_emb),
        "n_seeds": int(n_seeds),
        "res_list": [float(r) for r in res_list],
        "n_clusters": int(k_final),
        "consensus_backend": backend,
        "neighbors_backend": str(neighbors_backend),
        "scale_mode": str(scale_mode),
    }
    return y_robust_smooth


# ----------------------------------------------------------------------
# Mclust (R) + fallback to sklearn GMM
# ----------------------------------------------------------------------
def mclust_R(x, num_cluster, modelNames=None, random_state=0):
    if not HAS_RPY2:
        raise ImportError("rpy2 is not installed; cannot use Mclust.")

    if modelNames is None:
        modelNames = robjects.NULL

    robjects.r("set.seed")(random_state)

    r_mclust_func = robjects.r(
        """
        function(x, G, modelName){
            suppressMessages(library(mclust))
            res <- Mclust(x, G=G, modelNames=modelName, verbose=FALSE)
            res$classification
        }
        """
    )

    with numpy2ri.converter.context():
        r_res = r_mclust_func(x, num_cluster, modelNames)
        clu = np.array(r_res).astype(int)

    return clu - 1


import time

def cluster_with_mclust(Z: np.ndarray, k: int, random_state: int = 0, show_progress: bool = False) -> np.ndarray:
    def _log(msg):
        if show_progress:
            print(msg, flush=True)

    if HAS_RPY2:
        _log("[mclust] Using R::Mclust via rpy2... (this may take a while)")
        t0 = time.time()
        Z_use = np.asarray(Z, dtype=np.float64)

        _log("[mclust] Running Mclust() ...")
        labels = mclust_R(
            Z_use,
            num_cluster=int(k),
            modelNames=robjects.NULL,
            random_state=random_state,
        )
        _log(f"[mclust] Done. elapsed={time.time()-t0:.2f}s")
        return labels.astype(int)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _log("[mclust] rpy2 not available; fallback to sklearn GaussianMixture.")
    t0 = time.time()

    Z = np.asarray(Z, dtype=np.float64)

    # Stage 1: PCA (optional)
    if Z.shape[1] > 50:
        n_components = 50
        _log(f"[mclust] Stage 1/3: PCA {Z.shape[1]} -> {n_components} ...")
        t1 = time.time()
        pca = PCA(n_components=n_components, random_state=random_state)
        Z_red = pca.fit_transform(Z)
        _log(f"[mclust] PCA done. elapsed={time.time()-t1:.2f}s")
    else:
        _log(f"[mclust] Stage 1/3: skip PCA (dim={Z.shape[1]})")
        Z_red = Z

    # Stage 2: Standardize
    _log("[mclust] Stage 2/3: StandardScaler ...")
    t2 = time.time()
    Z_std = StandardScaler().fit_transform(Z_red)
    _log(f"[mclust] StandardScaler done. elapsed={time.time()-t2:.2f}s")

    # Stage 3: GMM EM
    cov_type = "full" if Z_std.shape[1] <= 32 else "diag"
    _log(f"[mclust] Stage 3/3: GaussianMixture (k={k}, cov='{cov_type}') EM fitting ...")

    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type="diag",
        random_state=random_state,
        n_init=50,
        max_iter=2000,
        reg_covar=1e-3,
        init_params="kmeans",
        verbose=1 if show_progress else 0,
        verbose_interval=50,
    )
    labels = gmm.fit_predict(Z_std)

    _log(f"[mclust] GMM done. total elapsed={time.time()-t0:.2f}s")
    return labels.astype(int)


# ----------------------------------------------------------------------
# Clustering dispatcher
# ----------------------------------------------------------------------
def cluster_with_method(
    adata,
    Z_spot: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 0,
    resolution: float | None = None,
    knn_k: int = 15,
    key_added: str = "cluster",
    use_rep: str | None = None,
    w_spa: float = 0.40,
    w_emb: float = 0.60,
    robust_seeds: int = 3,
    robust_smooth_iter: int = 1,
    robust_random_state: int = 0,
    robust_res_list: list[float] | None = None,
    robust_consensus_backend: str = "kmeans",
    robust_batch_size: int = 8192,
    neighbors_backend: str = "auto",
    neighbors_gpu_device: str = "cuda",
    scale_mode: str = "auto",
) -> np.ndarray:

    method = method.lower().strip()
    N = int(Z_spot.shape[0])
    scale_mode_eff = _resolve_scale_mode(N, scale_mode)

    if method == "kmeans":
        k = n_clusters if n_clusters and n_clusters > 0 else _estimate_k(N)
        kmeans_n_init = 10 if scale_mode_eff == "small" else 5
        print(f"[cluster] KMeans, n_clusters={k}, n_init={kmeans_n_init}")
        km = KMeans(n_clusters=k, n_init=kmeans_n_init, random_state=0)
        labels = km.fit_predict(Z_spot).astype(int)
        adata.obs[key_added] = labels
        adata.obs[key_added] = adata.obs[key_added].astype(int)
        adata.uns[f"{key_added}_params"] = {
            "method": "kmeans",
            "n_clusters": int(k),
            "n_init": int(kmeans_n_init),
            "scale_mode": str(scale_mode_eff),
        }
        return labels

    if method == "leiden":
        if use_rep is not None and use_rep in adata.obsm:
            _neighbors_from_rep(
                adata,
                use_rep=use_rep,
                n_neighbors=knn_k,
                backend=neighbors_backend,
                gpu_device=neighbors_gpu_device,
            )
        else:
            _neighbors_from_embedding(
                adata,
                Z_spot,
                knn_k=knn_k,
                backend=neighbors_backend,
                gpu_device=neighbors_gpu_device,
            )
        res = float(resolution) if resolution is not None else 1.0
        print(f"[cluster] Leiden, resolution={res}, knn_k={knn_k}")
        _run_leiden(adata, resolution=res, key_added=key_added)
        adata.obs[key_added] = adata.obs[key_added].astype(int)
        adata.uns[f"{key_added}_params"] = {
            "method": "leiden",
            "resolution": float(res),
            "knn_k": int(knn_k),
            "neighbors_backend": str(neighbors_backend),
            "scale_mode": str(scale_mode_eff),
        }
        return adata.obs[key_added].to_numpy()

    if method == "louvain":
        if use_rep is not None and use_rep in adata.obsm:
            _neighbors_from_rep(
                adata,
                use_rep=use_rep,
                n_neighbors=knn_k,
                backend=neighbors_backend,
                gpu_device=neighbors_gpu_device,
            )
        else:
            _neighbors_from_embedding(
                adata,
                Z_spot,
                knn_k=knn_k,
                backend=neighbors_backend,
                gpu_device=neighbors_gpu_device,
            )
        res = float(resolution) if resolution is not None else 1.0
        print(f"[cluster] Louvain, resolution={res}, knn_k={knn_k}")
        sc.tl.louvain(adata, resolution=res, key_added=key_added)
        adata.obs[key_added] = adata.obs[key_added].astype(int)
        adata.uns[f"{key_added}_params"] = {
            "method": "louvain",
            "resolution": float(res),
            "knn_k": int(knn_k),
            "neighbors_backend": str(neighbors_backend),
            "scale_mode": str(scale_mode_eff),
        }
        return adata.obs[key_added].to_numpy()

    if method == "robust":
        # 1) base resolution
        if resolution is not None:
            base_res = float(resolution)
        else:
            base_res = 0.6

        # 2) resolution list (user override > auto)
        if robust_res_list is not None and len(robust_res_list) > 0:
            # robust_res_list should already be a list of floats
            res_list_use = [max(0.05, float(x)) for x in robust_res_list]
        else:
            if scale_mode_eff == "small":
                res_list_use = [max(0.05, base_res + d) for d in (-0.1, 0.0, 0.1)]
            elif scale_mode_eff == "medium":
                res_list_use = [max(0.05, base_res + d) for d in (-0.05, 0.05)]
            else:
                res_list_use = [max(0.05, base_res + d) for d in (-0.05, 0.05)]

        print(
            f"[cluster] Robust consensus Leiden, knn_k={knn_k}, "
            f"base_resolution={base_res}, res_list={res_list_use}, "
            f"seeds={robust_seeds}, smooth_iter={robust_smooth_iter}, rs={robust_random_state}, "
            f"neighbors_backend={neighbors_backend}, scale_mode={scale_mode_eff}"
        )

        # 3) n_clusters (optional)
        if n_clusters and n_clusters > 0:
            n_clusters_final = int(n_clusters)
        else:
            n_clusters_final = None

        # 4) rep selection
        rep = use_rep
        if rep is None:
            rep = "X_emb" if "X_emb" in adata.obsm else "X"

        # NOTE: "X" is special; others must be in obsm
        if rep != "X" and rep not in adata.obsm:
            print(f"[warn] use_rep={rep} not in adata.obsm, fallback to X_emb/X")
            rep = "X_emb" if "X_emb" in adata.obsm else "X"

        # 5) store weights
        w_spa = float(w_spa)
        w_emb = float(w_emb)
        adata.uns["w_spa"] = w_spa
        adata.uns["w_emb"] = w_emb

        # 6) run robust consensus
        labels = robust_consensus_leiden(
            adata,
            use_rep=rep,
            k_emb=int(knn_k),
            key_added=key_added,
            w_spa=w_spa,
            w_emb=w_emb,
            base_resolution=base_res,
            res_list=res_list_use,
            n_clusters=n_clusters_final,
            n_seeds=int(robust_seeds),
            n_smooth_iter=int(robust_smooth_iter),
            random_state=int(robust_random_state),
            consensus_backend=robust_consensus_backend,
            consensus_batch_size=int(robust_batch_size),
            neighbors_backend=neighbors_backend,
            neighbors_gpu_device=neighbors_gpu_device,
            scale_mode=scale_mode_eff,
        )
        return labels

    if method == "mclust":
        k = n_clusters if n_clusters and n_clusters > 0 else _estimate_k(N)
        print(f"[cluster] Mclust/GMM, n_clusters={k}")

        rep = use_rep
        if rep is None:
            rep = "X_emb" if ("X_emb" in adata.obsm) else "X"

        if rep == "X":
            Z_use = adata.X
        elif rep in adata.obsm:
            Z_use = adata.obsm[rep]
        elif rep in adata.layers:
            Z_use = adata.layers[rep]
        else:
            print(f"[warn] use_rep='{rep}' not found in adata.obsm/adata.layers; fallback to Z_spot")
            Z_use = Z_spot

        labels = cluster_with_mclust(
            Z_use,
            k=int(k),
            random_state=0,
            show_progress=bool(adata.uns.get("show_progress", False)),
        )

        adata.obs[key_added] = labels
        adata.obs[key_added] = adata.obs[key_added].astype(int)

        adata.uns[f"{key_added}_params"] = {
            "method": "mclust",
            "n_clusters": int(k),
            "use_rep": rep,
        }
        return labels

    raise ValueError(
        f"Unknown method '{method}'. Supported: kmeans, leiden, louvain, robust, mclust."
    )


# ----------------------------------------------------------------------
# Metrics and utilities
# ----------------------------------------------------------------------
def hungarian_match(gt_labels, pred_labels):
    """
    Hungarian matching for cluster labels based on confusion matrix.
    """
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred_labels)
    classes = np.unique(gt_labels)
    clusters = np.unique(pred_labels)
    n_class = classes.size
    n_cluster = clusters.size
    cost_matrix = np.zeros((n_class, n_cluster), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cost_matrix[i, j] = np.sum((gt_labels == c) & (pred_labels == k))
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = {clusters[j]: classes[i] for i, j in zip(row_ind, col_ind)}
    new_pred = np.array([mapping.get(l, -1) for l in pred_labels])
    acc = accuracy_score(gt_labels, new_pred)
    return acc, mapping

def merge_small_clusters_by_centroid(
    labels: np.ndarray,
    Z: np.ndarray,
    min_size: int = 50,
    max_rounds: int = 10,
) -> np.ndarray:

    labels = np.asarray(labels, dtype=int).copy()
    Z = np.asarray(Z, dtype=np.float32)
    N = labels.shape[0]
    assert Z.shape[0] == N

    for _round in range(int(max_rounds)):
        uniq, counts = np.unique(labels, return_counts=True)
        sizes = dict(zip(uniq.tolist(), counts.tolist()))

        small = [c for c in uniq.tolist() if sizes[c] < int(min_size)]
        if len(small) == 0:
            break

        large = [c for c in uniq.tolist() if sizes[c] >= int(min_size)]
        if len(large) == 0:
            break

        centroids = {}
        for c in uniq.tolist():
            idx = (labels == c)
            if idx.sum() > 0:
                centroids[c] = Z[idx].mean(axis=0)

        for c in small:
            if c not in centroids:
                continue
            cc = centroids[c]
            best_t = None
            best_d = None
            for t in large:
                if t not in centroids:
                    continue
                d = float(np.sum((cc - centroids[t]) ** 2))  # squared euclid
                if (best_d is None) or (d < best_d):
                    best_d = d
                    best_t = t
            if best_t is not None:
                labels[labels == c] = best_t

    uniq = np.unique(labels)
    remap = {c: i for i, c in enumerate(uniq.tolist())}
    labels = np.array([remap[int(x)] for x in labels], dtype=int)
    return labels

def _get_rep_matrix(adata, rep, fallback_Z=None):
    """
    Return a dense float32 matrix for metric computation, following rep priority:
    - None / "X" -> adata.X
    - adata.obsm[rep]
    - adata.layers[rep]
    - fallback_Z (npz embedding) if provided
    """
    import numpy as np
    import scipy.sparse as sp

    if rep is None or str(rep).lower() in ("x", "adata.x"):
        X = adata.X
    elif rep in adata.obsm:
        X = adata.obsm[rep]
    elif rep in adata.layers:
        X = adata.layers[rep]
    else:
        if fallback_Z is None:
            raise KeyError(f"rep '{rep}' not found in adata.obsm/adata.layers and no fallback_Z provided.")
        X = fallback_Z

    # convert sparse -> dense
    if sp.issparse(X):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)


def compute_metrics(
    Z: np.ndarray,
    labels: np.ndarray,
    adata: ad.AnnData | None = None,
    label_key: str | None = None,
    calc_acc: bool = False,
):
    metrics = {}
    labels = np.asarray(labels, dtype=int)

    metrics["n_clusters"] = int(len(np.unique(labels)))

    # NMI / ARI / ACC
    if adata is not None and label_key is not None and label_key in adata.obs:
        gt = adata.obs[label_key].to_numpy()

        gt_str = np.asarray(gt, dtype=str)
        bad = {"na", "nan", "none", "null", "unknown", ""}
        valid = np.array([s.strip().lower() not in bad for s in gt_str], dtype=bool)
        print(f"[metrics] labeled={valid.sum()}/{len(valid)} (nan/na filtered)")


        if valid.sum() > 0:
            gt_valid = gt_str[valid]
            pred_valid = labels[valid]
            metrics["NMI"] = float(normalized_mutual_info_score(gt_valid, pred_valid))
            metrics["ARI"] = float(adjusted_rand_score(gt_valid, pred_valid))
            if calc_acc:
                acc, _ = hungarian_match(gt_valid, pred_valid)
                metrics["ACC"] = float(acc)
        else:
            metrics["NMI"] = float("nan")
            metrics["ARI"] = float("nan")
            if calc_acc:
                metrics["ACC"] = float("nan")
    else:
        metrics["NMI"] = float("nan")
        metrics["ARI"] = float("nan")
        if calc_acc:
            metrics["ACC"] = float("nan")

    # Silhouette / CH / DB
    if Z is not None and Z.shape[0] == labels.shape[0]:
        try:
            if len(np.unique(labels)) > 1:
                metrics["Silhouette"] = float(silhouette_score(Z, labels))
            else:
                metrics["Silhouette"] = float("nan")
        except Exception:
            metrics["Silhouette"] = float("nan")

        try:
            if len(np.unique(labels)) > 1:
                metrics["Calinski_Harabasz"] = float(calinski_harabasz_score(Z, labels))
            else:
                metrics["Calinski_Harabasz"] = float("nan")
        except Exception:
            metrics["Calinski_Harabasz"] = float("nan")

        try:
            if len(np.unique(labels)) > 1:
                metrics["Davies_Bouldin"] = float(davies_bouldin_score(Z, labels))
            else:
                metrics["Davies_Bouldin"] = float("nan")
        except Exception:
            metrics["Davies_Bouldin"] = float("nan")

    # cluster size stats
    counts = Counter(labels)
    sizes = np.array(list(counts.values()), dtype=float)
    metrics["largest_cluster_size"] = float(sizes.max())
    metrics["smallest_cluster_size"] = float(sizes.min())
    metrics["cluster_size_ratio_max_min"] = float(sizes.max() / max(1.0, sizes.min()))
    p = sizes / sizes.sum()
    entropy = -np.sum(p * np.log2(p + 1e-12))
    metrics["cluster_distribution_entropy"] = float(entropy)

    return metrics


def load_npz(npz_path: str):
    """
    More robust loader for *.for_cluster.npz:
      - Try common embedding keys: 'Z', 'embedding', 'emb', 'X_emb', 'feat', ...
      - If not found, fall back to first 2D float array.
      - Try common obs_names keys: 'obs_names', 'cell_id', 'cells', 'barcodes'.
      - Try common spatial keys: 'spatial', 'coords', 'xy', 'pos'.
    """
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.files)
    print(f"[load_npz] keys in {npz_path}: {keys}")

    # 1) embedding
    Z = None
    cand_Z = ["Z", "embedding", "emb", "X_emb", "feat", "features"]
    for k in cand_Z:
        if k in keys:
            Z = npz[k]
            print(f"[load_npz] use '{k}' as embedding, shape={Z.shape}")
            break
    if Z is None:
        for k in keys:
            arr = npz[k]
            if hasattr(arr, "ndim") and arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
                Z = arr
                print(f"[load_npz] fallback: use '{k}' as embedding, shape={Z.shape}")
                break
    if Z is None:
        raise KeyError("Cannot find embedding matrix in npz (tried keys: 'Z', 'embedding', 'emb', 'X_emb', ...).")

    # 2) obs_names
    obs_names = None
    cand_obs = ["obs_names", "cell_id", "cells", "barcodes"]
    for k in cand_obs:
        if k in keys:
            obs_names = np.asarray(npz[k]).astype(str)
            print(f"[load_npz] use '{k}' as obs_names, len={len(obs_names)}")
            break
    if obs_names is None:
        obs_names = np.array([f"cell_{i}" for i in range(Z.shape[0])], dtype=str)
        print(f"[load_npz] obs_names not found, fallback to range index, len={len(obs_names)}")

    # 3) spatial
    spatial = None
    cand_spatial = ["spatial", "coords", "xy", "pos"]
    for k in cand_spatial:
        if k in keys:
            arr = np.asarray(npz[k])
            if arr.ndim == 2 and arr.shape[0] == Z.shape[0]:
                spatial = arr
                print(f"[load_npz] use '{k}' as spatial, shape={spatial.shape}")
                break

    # 4) adjacency triplets
    A_row = npz.get("A_row", None)
    A_col = npz.get("A_col", None)
    A_weight = npz.get("A_weight", None)

    # 5) optional rep_key
    rep_key_np = npz.get("rep_key", None)
    if rep_key_np is not None:
        try:
            rep_key_np = str(rep_key_np.item())
        except Exception:
            rep_key_np = str(rep_key_np)

    return Z, obs_names, spatial, A_row, A_col, A_weight, rep_key_np


def build_adata(
    Z: np.ndarray,
    obs_names: np.ndarray,
    spatial: np.ndarray | None,
    A_row: np.ndarray | None,
    A_col: np.ndarray | None,
    A_weight: np.ndarray | None,
    rep_key: str | None = None,
) -> ad.AnnData:

    # store embedding as main X
    adata = ad.AnnData(X=Z.copy())
    adata.obs_names = obs_names.astype(str)

    # also store embedding in obsm so robust consensus can use it
    Z32 = np.asarray(Z, dtype=np.float32)

    # Always provide stable keys
    adata.obsm["X"] = Z32
    adata.obsm["emb"] = Z32

    # store under rep_key and common alias "emb"
    if rep_key is not None:
        rep_key = str(rep_key).strip()
        if rep_key != "":
            adata.obsm[rep_key] = Z32

    # many users pass --use_rep emb, so make sure it exists
    adata.obsm["emb"] = Z32

    if spatial is not None:
        if spatial.shape[0] != Z.shape[0]:
            print(
                f"[build_adata] WARNING: spatial.shape[0]={spatial.shape[0]} "
                f"!= Z.shape[0]={Z.shape[0]}, skip spatial."
            )
        else:
            adata.obsm["spatial"] = spatial[:, :2]

    if A_row is not None and A_col is not None and A_weight is not None:
        N = Z.shape[0]
        A = csr_matrix((A_weight, (A_row, A_col)), shape=(N, N))
        A = _ensure_undirected(A)
        A = A.tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        adata.obsp["A_spatial"] = A
    else:
        print("[build_adata] No adjacency in npz; A_spatial will be missing.")

    return adata

def plot_embedding_2d(Z, labels, out_path, title="Embedding 2D (PCA)"):
    if Z.shape[1] < 2:
        print("[plot_embedding_2d] Embedding dim < 2, skip 2D plot.")
        return
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)
    plt.figure(figsize=(6, 5))
    sc_plt = plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=5, cmap="tab20")
    plt.colorbar(sc_plt, label="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_spatial(spatial, labels, out_path, title="Spatial"):
    if spatial is None or spatial.shape[1] < 2:
        print("[plot_spatial] No valid spatial coordinates, skip spatial plot.")
        return
    x = spatial[:, 0]
    y = spatial[:, 1]
    plt.figure(figsize=(6, 5))
    sc_plt = plt.scatter(x, y, c=labels, s=5, cmap="tab20")
    # sc_plt = plt.scatter(x, y, c=labels, s=0.8, cmap="tab20")
    # sc_plt = plt.scatter(x, y, c=labels, s=0.4, cmap="tab20")


    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")   

    plt.colorbar(sc_plt, label="Cluster")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_labels(obs_names, labels, out_path):
    with open(out_path, "w") as f:
        f.write("cell_id\tcluster\n")
        for cid, lab in zip(obs_names, labels):
            f.write(f"{cid}\t{int(lab)}\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--npz", required=True, help="Path to *.for_cluster.npz")
    parser.add_argument(
        "--method",
        choices=["kmeans", "leiden", "louvain", "robust", "mclust"],
        default="kmeans",
    )
    parser.add_argument(
        "--scale_mode",
        choices=["auto", "small", "medium", "large"],
        default="auto",
        help="Budget preset by dataset size. 'auto' uses n_obs to choose small/medium/large.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=0,
        help="Number of clusters (0 = auto for some methods).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Resolution for Leiden/Louvain (default 1.0) and robust (default 0.6).",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=15,
        help="K for KNN graph on embedding.",
    )
    parser.add_argument(
        "--neighbors_backend",
        choices=["scanpy", "gpu", "auto"],
        default="auto",
        help="Neighbor graph backend: auto (default), scanpy, or gpu.",
    )
    parser.add_argument(
        "--neighbors_gpu_device",
        type=str,
        default="cuda",
        help="Torch device used by GPU neighbor backend (default: cuda).",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Output prefix (default: npz path without extension).",
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Optional path to original h5ad to compute NMI/ARI/ACC.",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default=None,
        help="obs column in h5ad used as ground truth label.",
    )
    parser.add_argument(
        "--calc_acc",
        action="store_true",
        help="If set, compute ACC via Hungarian matching.",
    )
    parser.add_argument(
        "--w_spa",
        type=float,
        default=0.4,
        help="Spatial weight for robust clustering (default 0.4).",
    )
    parser.add_argument(
        "--w_emb",
        type=float,
        default=0.6,
        help="Embedding weight for robust clustering (default 0.6).",
    )
    parser.add_argument(
        "--robust_seeds",
        type=int,
        default=3,
        help="Robust consensus seeds (base default: 3; auto-adjusted by --scale_mode if not explicitly set).",
    )
    parser.add_argument(
        "--robust_smooth_iter",
        type=int,
        default=1,
        help="Robust consensus smoothing iters (base default: 1; auto-adjusted by --scale_mode if not explicitly set).",
    )
    parser.add_argument(
        "--robust_random_state",
        type=int,
        default=0,
        help="Robust consensus: random_state (default: 0).",
    )
    parser.add_argument(
        "--robust_res_list",
        type=str,
        default=None,
        help=(
            "Robust consensus: comma-separated resolution list, e.g. "
            "'0.65,0.75,0.85,0.95'. If not set, auto list around --resolution."
        ),
    )
    parser.add_argument(
        "--robust_consensus_backend",
        choices=["kmeans", "minibatch"],
        default="kmeans",
        help="Backend for robust consensus meta clustering (base default: kmeans; auto-adjusted by --scale_mode if not explicitly set).",
    )
    parser.add_argument(
        "--robust_batch_size",
        type=int,
        default=8192,
        help="Batch size for robust MiniBatchKMeans (used when --robust_consensus_backend=minibatch).",
    )
    parser.add_argument(
        "--use_rep",
        type=str,
        default=None,
        help=(
            "Which representation to use for building the neighbor graph. "
            "If None, use adata.X. If set, it can be a key in adata.obsm (e.g. 'X_emb', 'X_pca') "
            "or a key in adata.layers."
        ),
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=0,
        help=(
            "If >0, apply PCA to the chosen representation before clustering. "
            "The PCA result is stored in adata.obsm[pca_key] and use_rep will be switched to pca_key."
        ),
    )
    parser.add_argument(
        "--pca_seed",
        type=int,
        default=0,
        help="Random seed for PCA (sklearn PCA random_state).",
    )
    parser.add_argument(
        "--pca_key",
        type=str,
        default="X_pca",
        help="Key name to store PCA representation in adata.obsm (default: X_pca).",
    )
    parser.add_argument(
        "--power",
        type=int,
        default=0,
        help="Graph propagation steps for 'power' embedding smoothing (0 = disable).",
    )
    parser.add_argument(
        "--power_alpha",
        type=float,
        default=1.0,
        help="Weight for smoothed features when power > 0, final = local + alpha * smoothed.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply spatial KNN smoothing on embedding before clustering.",
    )
    parser.add_argument(
        "--smooth_k",
        type=int,
        default=20,
        help="K for spatial KNN smoothing (default 20).",
    )
    parser.add_argument(
        "--smooth_iter",
        type=int,
        default=1,
        help="Iterations for spatial KNN smoothing (default 1).",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine cluster labels by spatial majority vote on A_spatial.",
    )
    parser.add_argument(
        "--refine_iter",
        type=int,
        default=1,
        help="Number of refinement iterations when --refine is set.",
    )
    parser.add_argument(
        "--refine_k",
        type=int,
        default=0,
        help="If >0, refine labels by spatial KNN majority vote with this k (uses coords).",
    )
    parser.add_argument(
        "--merge_small",
        action="store_true",
        help="Merge tiny clusters into nearest large cluster by embedding centroid.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=50,
        help="Minimum size threshold for clusters; smaller ones will be merged if --merge_small is set.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress messages for long steps (mclust/GMM/PCA).",
    )
    parser.add_argument(
        "--metrics_rep",
        type=str,
        default=None,
        help="Representation used to compute Silhouette/CH/DB. Default: same as --use_rep after PCA.",
    )

    args = parser.parse_args()

    parsed_robust_res_list = None
    if args.robust_res_list is not None:
        s = args.robust_res_list.strip()
        if s != "":
            parsed_robust_res_list = [float(x) for x in s.split(",") if x.strip() != ""]

    # Output prefix and folder
    if args.out_prefix is None:
        args.out_prefix = os.path.splitext(args.npz)[0]

    out_dir = os.path.join(os.path.dirname(args.out_prefix), "cluster_results")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(args.out_prefix)
    args.out_prefix = os.path.join(out_dir, base_name)

    w_spa = float(args.w_spa)
    w_emb = float(args.w_emb)

    print("Loading npz...")
    Z, obs_names, spatial, A_row, A_col, A_weight, rep_key_np = load_npz(args.npz)
    print(f"Embedding shape: {Z.shape}")
    print(f"[npz] rep_key = {rep_key_np}")

    print("Building AnnData...")
    adata = build_adata(Z, obs_names, spatial, A_row, A_col, A_weight, rep_key=rep_key_np)

    if args.h5 is not None and args.label_key is not None:
        print(f"Loading h5ad for labels: {args.h5}")
        try:
            adata_raw = sc.read_h5ad(args.h5)
            if args.label_key not in adata_raw.obs:
                print(
                    f"[warn] label_key '{args.label_key}' not found in h5ad.obs; "
                    "NMI/ARI/ACC will be skipped."
                )
            else:
                missing = np.setdiff1d(adata.obs_names, adata_raw.obs_names)
                if len(missing) > 0:
                    print(
                        f"[warn] {len(missing)} cells in npz not found in h5ad; "
                        "they will be assigned 'na' labels."
                    )
                common = adata.obs_names.intersection(adata_raw.obs_names)
                adata_raw = adata_raw[common, :].copy()
                adata.obs[args.label_key] = "na"
                adata.obs.loc[common, args.label_key] = (
                    adata_raw.obs[args.label_key].astype(str).values
                )
                print(f"Label '{args.label_key}' merged for {len(common)} cells.")
        except Exception as e:
            print(f"[warn] Failed to load/merge h5ad labels: {e}")

    # --- optional local feature smoothing (BEFORE clustering) ---
    if args.smooth:
        if spatial is None:
            print("[warn] --smooth requested but spatial is missing; skip smoothing.")
        else:
            print(
                f"Applying spatial KNN smoothing on embedding: "
                f"k={args.smooth_k}, iter={args.smooth_iter} ..."
            )
            Z = spatial_knn_smooth_embedding(Z, spatial, k=args.smooth_k, n_iter=args.smooth_iter)
            Z32 = np.asarray(Z, dtype=np.float32)
            adata.X = Z32
            adata.obsm["X"] = Z32
            adata.obsm["emb"] = Z32

    # Optional: Power Embedding smoothing (MAEST-like)
    if args.power > 0:
        if "A_spatial" not in adata.obsp:
            print(f"[warn] power={args.power} requested but A_spatial not found; skip power smoothing.")
        else:
            print(f"Applying power embedding smoothing: power={args.power}, alpha={args.power_alpha} ...")
            Z = power_smooth_embedding(Z, adata.obsp["A_spatial"], power=args.power, alpha=args.power_alpha)
            print("Power embedding done. New embedding shape:", Z.shape)
            Z32 = np.asarray(Z, dtype=np.float32)
            adata.X = Z32
            adata.obsm["X"] = Z32
            adata.obsm["emb"] = Z32

    adata.uns["show_progress"] = bool(args.progress)

    # -------------------------
    # Optional PCA on embedding
    # -------------------------
    if getattr(args, "pca_dim", 0):
        pca_dim = int(args.pca_dim)
        pca_key = getattr(args, "pca_key", "X_pca") or "X_pca"

        # Select the matrix to PCA from
        use_rep = getattr(args, "use_rep", None)
        if use_rep is None or str(use_rep).lower() in ("x", "adata.x"):
            X_for_pca = adata.X
        elif use_rep in adata.obsm:
            X_for_pca = adata.obsm[use_rep]
        elif use_rep in adata.layers:
            X_for_pca = adata.layers[use_rep]
        else:
            raise KeyError(
                f"--use_rep '{use_rep}' not found in adata.obsm or adata.layers. "
                "Use None/'X' to PCA adata.X, or pass a valid key (e.g. 'X_emb')."
            )

        # Convert to dense float32 for sklearn if needed
        if sp.issparse(X_for_pca):
            X_dense = X_for_pca.astype(np.float32).toarray()
        else:
            X_dense = np.asarray(X_for_pca, dtype=np.float32)

        n_obs, n_feat = X_dense.shape
        if pca_dim <= 0:
            raise ValueError("--pca_dim must be > 0.")
        if pca_dim >= n_feat:
            print(f"[PCA] skip: pca_dim={pca_dim} >= n_features={n_feat}.")
        else:
            print(f"Applying PCA on representation '{use_rep or 'X'}': {n_feat} -> {pca_dim} ...")
            pca = PCA(n_components=pca_dim, random_state=int(getattr(args, 'pca_seed', 0)))
            X_pca = pca.fit_transform(X_dense).astype(np.float32)

            # Store in obsm and switch use_rep for downstream steps
            adata.obsm[pca_key] = X_pca
            args.use_rep = pca_key

    scale_mode_eff = _resolve_scale_mode(int(adata.n_obs), args.scale_mode)

    if not _has_cli_flag("--knn_k"):
        args.knn_k = 15 if scale_mode_eff == "small" else (12 if scale_mode_eff == "medium" else 10)

    if not _has_cli_flag("--neighbors_backend"):
        args.neighbors_backend = "scanpy" if scale_mode_eff == "small" else "auto"

    if not _has_cli_flag("--robust_seeds"):
        args.robust_seeds = 5 if scale_mode_eff == "small" else (3 if scale_mode_eff == "medium" else 2)

    if not _has_cli_flag("--robust_smooth_iter"):
        args.robust_smooth_iter = 1 if scale_mode_eff in ("small", "medium") else 0

    if not _has_cli_flag("--robust_consensus_backend"):
        args.robust_consensus_backend = "kmeans" if scale_mode_eff in ("small", "medium") else "minibatch"

    print(
        f"[scale] mode={scale_mode_eff}, n_obs={adata.n_obs}, "
        f"knn_k={args.knn_k}, neighbors_backend={args.neighbors_backend}, "
        f"robust_seeds={args.robust_seeds}, robust_smooth_iter={args.robust_smooth_iter}, "
        f"robust_consensus_backend={args.robust_consensus_backend}"
    )

    print(f"Clustering with method={args.method}...")
    adata.uns["w_spa"] = w_spa
    adata.uns["w_emb"] = w_emb

    labels = cluster_with_method(
        adata, Z,
        method=args.method,
        n_clusters=args.n_clusters,
        resolution=args.resolution,
        knn_k=args.knn_k,
        key_added="cluster",
        use_rep=args.use_rep,
        w_spa=args.w_spa,
        w_emb=args.w_emb,
        robust_seeds=args.robust_seeds,
        robust_smooth_iter=args.robust_smooth_iter,
        robust_random_state=args.robust_random_state,
        robust_res_list=parsed_robust_res_list,
        robust_consensus_backend=args.robust_consensus_backend,
        robust_batch_size=args.robust_batch_size,
        neighbors_backend=args.neighbors_backend,
        neighbors_gpu_device=args.neighbors_gpu_device,
        scale_mode=scale_mode_eff,
    )

    adata.obs["cluster"] = labels.astype(int)

    # -------------------------
    # Prepare Z_metrics ONCE (use metrics_rep if set, else use_rep after PCA)
    # -------------------------
    rep_m = getattr(args, "metrics_rep", None) or getattr(args, "use_rep", None)
    try:
        Z_metrics = _get_rep_matrix(adata, rep_m, fallback_Z=Z)
    except Exception as e:
        print(f"[warn] failed to build Z_metrics from rep='{rep_m}', fallback to raw Z. err={e}")
        Z_metrics = np.asarray(Z, dtype=np.float32)

    print(f"[metrics] using rep='{rep_m}' with shape={getattr(Z_metrics, 'shape', None)}")

    print("Computing clustering metrics...")
    metrics = compute_metrics(
        Z_metrics,
        labels,
        adata=adata,
        label_key=args.label_key,
        calc_acc=args.calc_acc,
    )

    # Optional: spatial refinement (majority vote on A_spatial)
    if args.refine and "A_spatial" in adata.obsp:
        print(f"Refining labels by spatial majority vote on A_spatial (iter={args.refine_iter}) ...")
        labels = mrf_majority_smooth(
            adata.obsp["A_spatial"],
            labels,
            n_iter=int(args.refine_iter),
        ).astype(int)
        adata.obs["cluster"] = labels

        print("Recomputing clustering metrics after refinement...")
        metrics = compute_metrics(
            Z_metrics,
            labels,
            adata=adata,
            label_key=args.label_key,
            calc_acc=args.calc_acc,
        )

    # NEW: Optional refinement using spatial KNN by coordinates
    if args.refine_k and int(args.refine_k) > 0:
        if spatial is None:
            print("[warn] --refine_k requested but spatial is missing; skip refine_k.")
        else:
            print(f"Refining labels by spatial KNN majority vote (k={args.refine_k}, iter={args.refine_iter}) ...")
            labels = spatial_knn_refine_labels(
                labels,
                spatial=spatial,
                k=int(args.refine_k),
                n_iter=int(args.refine_iter),
            ).astype(int)
            adata.obs["cluster"] = labels

            print("Recomputing clustering metrics after spatial KNN refinement...")
            metrics = compute_metrics(
                Z_metrics,
                labels,
                adata=adata,
                label_key=args.label_key,
                calc_acc=args.calc_acc,
            )
    # --- merge tiny clusters (do this AFTER refine/refine_k) ---
    if args.merge_small:
        print(f"Merging tiny clusters: min_cluster_size={args.min_cluster_size} ...")

        # Use the SAME representation as clustering for merging
        rep = getattr(args, "use_rep", None)
        if rep is None or str(rep).lower() in ("x", "adata.x"):
            Z_merge = np.asarray(adata.X, dtype=np.float32)
        elif rep in adata.obsm:
            Z_merge = np.asarray(adata.obsm[rep], dtype=np.float32)
        elif rep in adata.layers:
            Z_merge = np.asarray(adata.layers[rep], dtype=np.float32)
        else:
            print(f"[warn] merge_small: use_rep='{rep}' not found; fallback to raw Z")
            Z_merge = np.asarray(Z, dtype=np.float32)

        labels = merge_small_clusters_by_centroid(
            labels,
            Z_merge,
            min_size=int(args.min_cluster_size),
            max_rounds=10,
        ).astype(int)
        adata.obs["cluster"] = labels

        print("Recomputing clustering metrics after merging tiny clusters...")
        metrics = compute_metrics(
            Z_metrics,
            labels,
            adata=adata,
            label_key=args.label_key,
            calc_acc=args.calc_acc,
        )

    # Save metrics
    metrics_path = args.out_prefix + f".{args.method}.metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}\t{v}\n")

    def _fmt(v):
        """Pretty float formatter for printing."""
        try:
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        except Exception:
            return str(v)

    print("--------------------------------------------------")
    print("📊 Clustering Metrics Summary")
    print(f"Number of clusters: {_fmt(metrics.get('n_clusters'))}")
    print(f"NMI: {_fmt(metrics.get('NMI'))}")
    print(f"ARI: {_fmt(metrics.get('ARI'))}")
    if "ACC" in metrics:
        print(f"ACC: {_fmt(metrics.get('ACC'))}")
    print(f"Silhouette Score: {_fmt(metrics.get('Silhouette'))}")
    print(f"Calinski–Harabasz: {_fmt(metrics.get('Calinski_Harabasz'))}")
    print(f"Davies–Bouldin: {_fmt(metrics.get('Davies_Bouldin'))}")
    print(f"Largest cluster size: {_fmt(metrics.get('largest_cluster_size'))}")
    print(f"Smallest cluster size: {_fmt(metrics.get('smallest_cluster_size'))}")
    print("Cluster size ratio (max/min): " f"{_fmt(metrics.get('cluster_size_ratio_max_min'))}")
    print("Cluster distribution entropy: " f"{_fmt(metrics.get('cluster_distribution_entropy'))}")
    print("--------------------------------------------------")
    print(f"\nMetrics saved to: {metrics_path}")

    labels_path = args.out_prefix + f".{args.method}.labels.txt"
    save_labels(obs_names, labels, labels_path)
    print(f"Labels saved to: {labels_path}")

    emb_fig_path = args.out_prefix + f".{args.method}.UMAP_2d.png"
    print("Computing 2D projection...")
    plot_embedding_2d(Z_metrics, labels, emb_fig_path, title=f"2D embedding ({args.method})")
    print(f"Embedding plot saved to: {emb_fig_path}")

    spatial_fig_path = args.out_prefix + f".{args.method}.spatial.png"
    plot_spatial(spatial, labels, spatial_fig_path, title=f"Spatial ({args.method})")
    print(f"Spatial plot saved to: {spatial_fig_path}")

if __name__ == "__main__":
    main()
