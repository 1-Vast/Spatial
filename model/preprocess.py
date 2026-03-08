from typing import List, Optional, Union
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from scipy.sparse.linalg import eigsh

def Cal_Spatial_Net(
    adata: AnnData,
    rad_cutoff: Optional[float] = None,
    k_cutoff: Optional[int] = None,
    model: str = "KNN",
    return_data: bool = False,
    graph_dir: Optional[str] = None,
    save_graph_csv: bool = True,
):

    print("🚀 Calculating spatial neighbor graph (GPU-accelerated) ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coords = torch.tensor(adata.obsm["spatial"], dtype=torch.float32, device=device)
    n_cells = adata.shape[0]

    if model == "KNN":
        num_workers = 0 if device == "cuda" else min(os.cpu_count(), 16)
        edge_index = knn_graph(
            x=coords,
            k=k_cutoff,
            loop=False,
            flow="target_to_source",
            num_workers=num_workers,
        )
    elif model == "Radius":
        num_workers = 0 if device == "cuda" else min(os.cpu_count(), 16)
        edge_index = radius_graph(
            x=coords,
            r=rad_cutoff,
            loop=True,
            flow="target_to_source",
            num_workers=num_workers,
        )
    else:
        raise ValueError("❌ Unknown model type. Use 'KNN' or 'Radius'.")

    edge_index = edge_index.cpu()
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    G = pd.DataFrame({"Cell1": row, "Cell2": col})

    G["Cell1"] = G["Cell1"].astype(int)
    G["Cell2"] = G["Cell2"].astype(int)
    adata.uns["Spatial_Net"] = G

    if not save_graph_csv:
        print("Skip saving Spatial_Net CSV.")
    else:
        if graph_dir is not None:
            base_dir = graph_dir
            base_name = (
                os.path.basename(adata.filename).split(".h5ad")[0]
                if getattr(adata, "filename", None)
                else "adata"
            )
        else:
            if getattr(adata, "filename", None):
                base_dir = os.path.dirname(adata.filename)
                base_name = os.path.basename(adata.filename).split(".h5ad")[0]
            elif "source" in adata.uns:
                base_dir = os.path.dirname(adata.uns["source"])
                base_name = os.path.basename(adata.uns["source"]).split(".h5ad")[0]
            else:
                base_dir = "."
                base_name = "adata"

            base_dir = os.path.join(base_dir, "SpatialGraphs")

        os.makedirs(base_dir, exist_ok=True)
        csv_path = os.path.join(base_dir, f"{base_name}_spatial_net.csv")
        G.to_csv(csv_path, index=False)
        print(f"💾 Spatial graph saved to: {csv_path}")

    print(f"✅ The graph contains {edge_index.shape[1]} edges, {n_cells} cells.")
    print(f"Average neighbors per cell: {edge_index.shape[1] / n_cells:.2f}")

    if return_data:
        return adata

def compute_pseudo_layer_ids_from_spatial(
    spatial,
    n_bins: int,
    knn_k: int = 18,
    mode: str = "fiedler",
    binning: str = "quantile",
    seed: int = 0,
):
    """
    Build pseudo-layer ids from spatial coordinates using graph Laplacian only.
    Only `mode='fiedler'` is supported to enforce spectral pseudo-depth.
    """

    import numpy as np

    spatial = np.asarray(spatial)
    assert spatial.ndim == 2 and spatial.shape[1] == 2, "spatial must be (N,2)"
    N = spatial.shape[0]
    if n_bins <= 1:
        return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.float32)

    if mode.lower() != "fiedler":
        raise ValueError(f"Unsupported mode={mode}. Use mode='fiedler'.")

    # ---- 1) build spatial kNN graph (undirected) ----
    # torch_geometric.nn.knn_graph returns directed edges; we symmetrize later.
    import torch
    from torch_geometric.nn import knn_graph

    x = torch.tensor(spatial, dtype=torch.float32)
    edge_index = knn_graph(x, k=int(knn_k), loop=False)  # (2, E)
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()

    # Symmetrize adjacency
    row2 = np.concatenate([row, col])
    col2 = np.concatenate([col, row])
    data = np.ones_like(row2, dtype=np.float32)
    A = sp.csr_matrix((data, (row2, col2)), shape=(N, N))
    A.data[:] = 1.0

    # ---- 2) graph Laplacian L = D - A ----
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    D = sp.diags(deg, format="csr")
    L = (D - A).astype(np.float64)

    # ---- 3) Fiedler vector (2nd smallest eigenvector) ----
    # Smallest eigenvector is constant; we take the 2nd.
    # eigsh may return unsorted; we sort by eigenvalue.
    vals, vecs = eigsh(L, k=2, which="SM")
    order = np.argsort(vals)
    fiedler = vecs[:, order[1]].astype(np.float32)

    # Normalize to a stable 0~1 pseudo-depth (rank-based is more robust)
    # This also removes sign ambiguity of eigenvectors.
    ranks = np.argsort(np.argsort(fiedler))
    pseudo_depth = ranks.astype(np.float32) / max(1, N - 1)

    # ---- 4) binning into pseudo layers ----
    if binning.lower() == "quantile":
        # equal number of points per bin (robust to density variation)
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(pseudo_depth, qs)
        # avoid identical edges causing empty bins
        edges[0] -= 1e-6
        edges[-1] += 1e-6
    elif binning.lower() == "uniform":
        edges = np.linspace(pseudo_depth.min() - 1e-6, pseudo_depth.max() + 1e-6, n_bins + 1)
    else:
        raise ValueError(f"Unsupported binning={binning}. Use 'quantile' or 'uniform'.")

    layer_ids = np.digitize(pseudo_depth, edges[1:-1], right=False).astype(np.int64)
    layer_ids = np.clip(layer_ids, 0, n_bins - 1)

    return layer_ids, pseudo_depth


def scanpy_workflow(
    adata: AnnData,
    filter_cell: Optional[bool] = False,
    min_gene: Optional[int] = 200,
    min_cell: Optional[int] = 30,
    call_hvg: Optional[bool] = True,
    n_top_genes: Optional[Union[int, List]] = 2500,
    batch_key: Optional[str] = None,
    n_comps: Optional[int] = 50,
    viz: Optional[bool] = True,
    resolution: Optional[float] = 0.8,
) -> AnnData:

    # --- Step 1: select starting matrix (prefer raw counts) ---
    start_from_counts = False
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        start_from_counts = True
    elif getattr(adata, "raw", None) is not None:
        adata = adata.raw.to_adata()
        start_from_counts = True
    else:
        if looks_like_counts(adata.X):
            start_from_counts = True
        else:
            start_from_counts = False

    # --- Step 2: optional filtering ---
    if filter_cell:
        sc.pp.filter_cells(adata, min_genes=min_gene)
        sc.pp.filter_genes(adata, min_cells=min_cell)

    # --- Step 3: highly variable genes selection ---
    if call_hvg:
        if isinstance(n_top_genes, int):
            if adata.n_vars > n_top_genes:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=n_top_genes,
                    flavor="seurat_v3",
                    batch_key=batch_key,
                    layer=None,
                )
            else:
                adata.var["highly_variable"] = True
                print("All genes are marked as highly variable.")
        elif isinstance(n_top_genes, list):
            adata.var["highly_variable"] = False
            n_top_genes = list(set(adata.var.index).intersection(set(n_top_genes)))
            adata.var.loc[n_top_genes, "highly_variable"] = True
    else:
        print("Skipping highly variable gene selection.")

    # --- Step 3.5: subset to HVGs ---
    if call_hvg and "highly_variable" in adata.var:
        hv = adata.var["highly_variable"].values
        if hv.sum() > 0 and hv.sum() < adata.n_vars:
            adata = adata[:, hv].copy()
            print(f"✅ Subset to HVGs: {hv.sum()} genes")

    # --- Step 4: detect if negative ---
    try:
        has_neg = (adata.X.min() < 0) if sp.issparse(adata.X) else (np.min(adata.X) < 0)
    except Exception:
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        has_neg = (np.min(X) < 0)

    # --- Step 5: normalize/log1p ONCE for count-like data ---
    is_counts_like = False
    if not has_neg:
        try:
            is_counts_like = bool(start_from_counts or looks_like_counts(adata.X))
        except Exception:
            is_counts_like = bool(start_from_counts)

    if (not has_neg) and is_counts_like:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("⚠️ Skip normalize/log1p. has_neg=", has_neg, " start_from_counts=", start_from_counts,
              " looks_like_counts=", is_counts_like)

    # --- Step 6: convert sparse to dense and clean NaN/Inf ---
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 7: scale once only ---
    sc.pp.scale(adata, max_value=10)

    # --- Step 8: PCA with NaN cleanup ---
    if n_comps > 0:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")

    # --- Step 9: optional visualization and clustering ---
    if viz:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_comps)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)

    return adata

def looks_like_counts(X, max_check=200000):
    import numpy as np
    if hasattr(X, "toarray"):
        Xs = X[:min(X.shape[0], 2000), :min(X.shape[1], 2000)].toarray()
    else:
        Xs = np.asarray(X[:min(X.shape[0], 2000), :min(X.shape[1], 2000)])
    if Xs.min() < 0:
        return False
    frac = np.abs(Xs - np.round(Xs))
    return np.quantile(frac, 0.999) < 1e-3

def compute_pseudo_layer_ids(
    A, spatial, bins: int = 8,
    method: str = "fiedler",
    axis: int = 1,
):
    if str(method).lower() != "fiedler":
        raise ValueError(f"Unsupported method={method}. Use method='fiedler'.")

    A_sp = A.tocsr().astype(np.float32)
    A_sp = A_sp.maximum(A_sp.T)  # make undirected

    deg = np.asarray(A_sp.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    L = sp.eye(A_sp.shape[0], dtype=np.float32) - D_inv_sqrt @ A_sp @ D_inv_sqrt

    vals, vecs = eigsh(L, k=2, which="SM")
    depth = vecs[:, 1].astype(np.float32)

    dmin, dmax = float(depth.min()), float(depth.max())
    denom = (dmax - dmin) if (dmax > dmin) else 1.0
    depth01 = (depth - dmin) / denom
    ids_np = np.clip((depth01 * bins).astype(np.int64), 0, bins - 1)
    return ids_np

def compute_local_normals_from_spatial(spatial: np.ndarray, knn_k: int = 10):

    spatial = np.asarray(spatial, dtype=np.float32)
    assert spatial.ndim == 2 and spatial.shape[1] == 2

    N = spatial.shape[0]

    # brute-force knn (N=3460 ok). If you prefer, replace with faiss/pyg knn.
    # Compute squared distances
    d2 = ((spatial[:, None, :] - spatial[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)

    knn_idx = np.argpartition(d2, kth=knn_k, axis=1)[:, :knn_k]  # (N, k)

    normals = np.zeros((N, 2), dtype=np.float32)
    scales  = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        nb = spatial[knn_idx[i]]  # (k,2)
        X = nb - nb.mean(axis=0, keepdims=True)  # center
        C = (X.T @ X) / max(1, X.shape[0])       # 2x2 covariance

        # eigen-decomposition (2D small)
        vals, vecs = np.linalg.eigh(C)
        n = vecs[:, np.argmin(vals)]             # smallest eigenvector = normal
        n = n / (np.linalg.norm(n) + 1e-12)

        # robust scale along normal direction
        proj = np.abs((nb - spatial[i]) @ n)     # (k,)
        s = np.median(proj) + 1e-6

        normals[i] = n.astype(np.float32)
        scales[i]  = float(s)

    return normals, scales
