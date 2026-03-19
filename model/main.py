import argparse, os
import numpy as np
import scipy.sparse as sp
import anndata as ad
import torch
import re
from tqdm import tqdm
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from model.augment import build_augmented_graph
from model.preprocess import (
    Cal_Spatial_Net,
    scanpy_workflow,
    compute_pseudo_layer_ids_from_spatial,
    compute_local_normals_from_spatial
)
from model.encoder import (
    scipy_to_torch_coo,
    normalize_rows_sparse_torch,
    CollaborativeEncoder,
    CrossLayerScorer,
    sample_negatives,
    node_embedding_from_layers,
    FeatureDecoder,
)


def set_seed(seed: int = 0, strict: bool = False):
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic behavior for reproducible experiments.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        if strict:
            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    if strict and torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        print(
            "Strict deterministic mode enabled without CUBLAS_WORKSPACE_CONFIG. "
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch for CuBLAS determinism."
        )


def unique_trainable_params(*modules):
    seen = set()
    params = []
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            if p is None or not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(p)
    return params

def _infer_layer_key(adata, user_layer_key, allow_fallback: bool = True):

    if user_layer_key is not None and str(user_layer_key).lower() not in ["none", "null", ""]:
        if user_layer_key in adata.obs:
            return user_layer_key
        return None

    if not allow_fallback:
        return None

    for k in ["sce.layer_guess", "layer_guess", "Ground Truth", "spatial_domain"]:
        if k in adata.obs:
            return k
    return None

def _to_ordinal_layer_ids(series):
    """
    Convert layer labels to ordinal ids (0..L-1) with a stable order.
    Works for labels like 'Layer1', 'Layer2', ..., 'WM'.
    """
    def order_key(x):
        s = str(x)
        sl = s.lower()
        if sl in ["wm", "white matter"]:
            return 10_000
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
        return 9_999

    uniq = sorted(series.astype(str).unique().tolist(), key=order_key)
    mapping = {name: idx for idx, name in enumerate(uniq)}
    ids = series.astype(str).map(mapping).to_numpy()
    return ids, mapping


def get_A_from_spatial_net(adata):
    G = adata.uns["Spatial_Net"].copy()

    if np.issubdtype(G["Cell1"].dtype, np.integer) and np.issubdtype(
        G["Cell2"].dtype, np.integer
    ):
        row, col = G["Cell1"].to_numpy(), G["Cell2"].to_numpy()
    else:
        cells = np.array(adata.obs_names)
        idx = dict(zip(cells, range(cells.shape[0])))
        G["Cell1"] = G["Cell1"].map(idx)
        G["Cell2"] = G["Cell2"].map(idx)
        row, col = G["Cell1"].to_numpy(), G["Cell2"].to_numpy()

    A = sp.coo_matrix(
        (np.ones(G.shape[0], dtype=np.float32), (row, col)),
        shape=(adata.n_obs, adata.n_obs),
        dtype=np.float32,
    )
    return A.tocsr()


def _apply_spot_feature_mask(
    base_input: torch.Tensor,
    num_spot: int,
    mask_idx: torch.Tensor,
    mode: str = "zero",
    noise_std: float = 0.02,
) -> torch.Tensor:
    if mask_idx is None or mask_idx.numel() == 0:
        return base_input

    mode = str(mode).lower()
    masked = base_input.clone()
    spot_feat = masked[:num_spot]

    if mode == "zero":
        spot_feat[mask_idx] = 0.0
    elif mode == "noise":
        spot_feat[mask_idx] = torch.randn_like(spot_feat[mask_idx]) * float(noise_std)
    elif mode == "mean":
        mean_feat = spot_feat.mean(dim=0, keepdim=True)
        spot_feat[mask_idx] = mean_feat.expand(mask_idx.numel(), -1)
    else:
        raise ValueError(f"Unsupported feat_mask_mode={mode}. Use one of: zero/noise/mean.")

    masked[:num_spot] = spot_feat
    return masked


def main(a):
    set_seed(a.seed, strict=getattr(a, "strict_determinism", False))
    adata = ad.read_h5ad(a.h5)

    # ---------- output prefix (avoid overwrite) ----------
    if a.out_prefix and a.out_prefix.strip():
        prefix = a.out_prefix.strip()
    else:
        prefix = os.path.splitext(a.h5)[0]

    # ensure output dir exists (if prefix includes a directory)
    out_dir = os.path.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # if the target files already exist, append timestamp to avoid overwrite
    npz_test = f"{prefix}.augK{a.K}_d{a.dim}_for_cluster.npz"
    h5_test  = f"{prefix}.augK{a.K}_d{a.dim}_none.h5ad"
    if os.path.exists(npz_test) or os.path.exists(h5_test):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_{ts}"


    # ---------- spatial graph ----------
    if "Spatial_Net" in adata.uns:
        print("Spatial_Net already exists, skip graph construction.")
    else:
        adata = Cal_Spatial_Net(
            adata,
            rad_cutoff=a.radius,
            k_cutoff=a.k,
            model=a.graph_model,
            return_data=True,
            save_graph_csv=False,
        )

    # ---------- gene expression features (optional Scanpy workflow) ----------
    if a.use_scanpy_workflow:
        print("Running scanpy_workflow...")
        adata_p = scanpy_workflow(
            adata.copy(),
            n_comps=a.pca_comps,
            call_hvg=True,
        )
        if "X_pca" in adata_p.obsm:
            X = adata_p.obsm["X_pca"][:, :a.pca_comps]
            adata.obsm["X_pca"] = np.asarray(X, dtype=np.float32)
        else:
            print("ERROR: X_pca not found after scanpy_workflow.")
    else:
        print(f"Checking raw X data: {adata.X}")
        X = adata.X

    # Check if X is None or valid
    if X is None:
        print("ERROR: X is None, check the source of the data.")
    else:
        print(f"X shape: {X.shape}")

    # ---------- features to reconstruct (for MAE) ----------
    X_recon = None
    if a.lambda_recon > 0:
        if str(getattr(a, "recon_target", "pca")).lower() != "pca":
            raise ValueError(f"Unsupported recon_target={a.recon_target}. Only 'pca' is supported.")
        if a.use_scanpy_workflow:
            X_recon = torch.tensor(
                np.asarray(X, dtype=np.float32),
                dtype=torch.float32,
                device=a.device,
            )
            print(f"MAE: use PCA features dim={X_recon.shape[1]} as reconstruction target.")
        else:
            print("Warning: recon_target='pca' requires --use_scanpy_workflow. Disable MAE reconstruction.")
            a.lambda_recon = 0.0

    # ---------- adjacency matrix ----------
    if hasattr(adata, "obsp") and ("A_spatial" in getattr(adata, "obsp", {})):
        A = adata.obsp["A_spatial"].tocsr()
    elif "Spatial_Net" in adata.uns_keys():
        A = get_A_from_spatial_net(adata)
    else:
        Cal_Spatial_Net(
            adata,
            model=a.graph_model,
            rad_cutoff=a.radius,
            k_cutoff=a.k,
            save_graph_csv=False,
        )
        A = get_A_from_spatial_net(adata)

    A = A.tocsr().astype(np.float32)
    if sp.issparse(X):
        X_graph = X.tocsr().astype(np.float32)
    else:
        X_graph = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    spatial_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    # -------- optional layer ids (generic + configurable) --------
    layer_key = _infer_layer_key(adata, a.layer_key, allow_fallback=(not a.no_layer_fallback))
    layer_ids_np = None
    layer_mapping = None
    pseudo_depth_cont = None

    if layer_key is not None:
        layer_ids_np, layer_mapping = _to_ordinal_layer_ids(adata.obs[layer_key])
    elif a.layer_aware and (getattr(a, "pseudo_layer_bins", 0) > 0):
        layer_ids_np, pseudo_depth_cont = compute_pseudo_layer_ids_from_spatial(
            spatial_np,
            n_bins=int(a.pseudo_layer_bins),
            knn_k=int(getattr(a, "pseudo_layer_knn", 18)),
            mode="fiedler",
            binning=str(getattr(a, "pseudo_layer_binning", "quantile")),
            seed=int(getattr(a, "seed", 0)),
        )

    if pseudo_depth_cont is not None:
        spatial_for_layer = np.asarray(pseudo_depth_cont, dtype=np.float32).reshape(-1, 1)
        layer_axis_used = 0
    else:
        spatial_for_layer = spatial_np
        layer_axis_used = int(a.layer_axis)

    # ---------- augmented graph ----------
    Pe = build_augmented_graph(
        A, X_graph,
        alpha=a.alpha,
        normalize_A=True,
        normalize_X=True,
        topN=a.topN,
        spatial=spatial_for_layer,
        layer_aware=a.layer_aware,
        layer_gamma=a.layer_gamma,
        layer_axis=layer_axis_used,
        layer_scale_mode=a.layer_scale_mode,
        layer_cutoff=a.layer_cutoff,
    )


    P = scipy_to_torch_coo(Pe).to(a.device)
    Ptilde = normalize_rows_sparse_torch(P)

    init_embed = None
    N_spot = A.shape[0]
    n = P.size(0)

    if a.use_scanpy_workflow:
        X_np = np.asarray(X, dtype=np.float32)  # PCA features when scanpy_workflow
        feat_dim = X_np.shape[1]  # e.g. 96

        init_embed = torch.randn(n, feat_dim, device=a.device) * 0.02
        init_embed[:N_spot, :] = torch.tensor(
            X_np, dtype=torch.float32, device=a.device
        )

        if feat_dim == a.dim:
            print(
                f"Use X_pca to initialize first {N_spot} nodes, remaining {n - N_spot} nodes are random."
            )
        else:
            print(
                f"Use X_pca(dim={feat_dim}) to initialize first {N_spot} nodes, remaining {n - N_spot} nodes are random; "
                f"encoder will project {feat_dim}->{a.dim}."
            )
    else:
        print("Scanpy workflow is not used, node embeddings are randomly initialized.")
        init_embed = None

    # ---------- model and optimizer ----------
    n = P.size(0)

    # Helper function to get activation module
    def get_activation(name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'prelu':
            return nn.PReLU()
        elif name.lower() == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    activation_module = get_activation(a.activation)

    enc = CollaborativeEncoder(
        num_nodes=n,
        dim=a.dim,
        K=a.K,
        init_embed=init_embed,
        activation=activation_module,
    ).to(a.device)
    scorer = CrossLayerScorer(dim=a.dim, K=a.K, hidden=a.hidden, activation=activation_module).to(a.device)

    feat_decoder = None
    recon_crit = None
    if (X_recon is not None) and (a.lambda_recon > 0):
        if a.embed_agg == "concat":
            dec_in_dim = a.dim * (a.K + 1)
        else:
            dec_in_dim = a.dim
        feat_out_dim = X_recon.shape[1]
        feat_decoder = FeatureDecoder(
            in_dim=dec_in_dim,
            out_dim=feat_out_dim,
            hidden=256,
            activation=activation_module,
        ).to(a.device)
        recon_crit = torch.nn.MSELoss(reduction="mean")
        print(
            f"Enable MAE feature decoder: in_dim={dec_in_dim}, out_dim={feat_out_dim}, "
            f"lambda_recon={a.lambda_recon}, mask_ratio_feat={a.mask_ratio_feat}, "
            f"feat_mask_mode={a.feat_mask_mode}, feat_noise_std={a.feat_noise_std}, "
            f"recon_target={a.recon_target}."
        )

    params = unique_trainable_params(enc, scorer, feat_decoder)

    opt = torch.optim.Adam(
        params,
        lr=min(a.lr, 8e-4),
        weight_decay=a.weight_decay,
        betas=(0.9, 0.999),
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # Learning rate scheduler (鍙€?
    scheduler = None
    if a.scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        print("Using Cosine Annealing LR scheduler.")

    # -------- optional layer ids / normal-aware priors --------
    layer_ids = None
    if layer_ids_np is not None:
        layer_ids = torch.tensor(layer_ids_np, dtype=torch.long, device=a.device)

    normals = None
    normal_scales = None
    if getattr(a, "normal_aware", False):
        normals, normal_scales = compute_local_normals_from_spatial(
            spatial_np,
            knn_k=int(getattr(a, "normal_knn", 10)),
        )
        print(f"Normal-aware enabled. normal_knn={int(getattr(a, 'normal_knn', 10))}, "
              f"normal_margin={float(getattr(a, 'normal_margin', 1.0))}, "
              f"normal_gamma={float(getattr(a, 'normal_gamma', 2.0))}")

    if layer_key is not None:
        if a.layer_aware:
            print(
                "Layer-aware negatives: "
                f"source=obs, gamma={float(a.layer_gamma):.3f}, "
                f"margin={int(a.neg_layer_margin) if hasattr(a, 'neg_layer_margin') else 2}, "
                f"neg_adj_ratio={float(getattr(a, 'neg_adj_ratio', 0.3)):.2f}, "
                f"hard_ratio={float(getattr(a, 'neg_hard_ratio', 0.7)):.2f}, "
                f"oversample={int(getattr(a, 'neg_oversample', 6))}"
            )
        else:
            print(f"Layer key found ('{layer_key}') but layer_aware=False; use generic negatives.")
    elif pseudo_depth_cont is not None:
        print(
            "Layer-aware negatives: "
            f"source=pseudo_fiedler, bins={int(a.pseudo_layer_bins)}, "
            f"knn={int(getattr(a, 'pseudo_layer_knn', 18))}, "
            f"binning={str(getattr(a, 'pseudo_layer_binning', 'quantile'))}, "
            f"gamma={float(a.layer_gamma):.3f}, "
            f"margin={int(a.neg_layer_margin) if hasattr(a, 'neg_layer_margin') else 2}, "
            f"neg_adj_ratio={float(getattr(a, 'neg_adj_ratio', 0.3)):.2f}, "
            f"hard_ratio={float(getattr(a, 'neg_hard_ratio', 0.7)):.2f}, "
            f"oversample={int(getattr(a, 'neg_oversample', 6))}, "
            "layer_axis=0"
        )

    # ---------- training loop ----------
    pos_idx = P.indices().t()
    for ep in tqdm(range(a.epochs), desc="Training Epochs", ncols=100):
        enc.train()
        scorer.train()

        # positive samples
        take = min(a.pos_per_epoch, pos_idx.size(0))
        perm = torch.randperm(pos_idx.size(0), device=a.device)[:take]
        pos = pos_idx[perm]

        # MAE-style masking on encoder input (spot nodes only), then encode once
        mask_idx = None
        base_input_override = None
        if (feat_decoder is not None) and (recon_crit is not None):
            num_mask = max(1, int(a.mask_ratio_feat * N_spot))
            mask_idx = torch.randperm(N_spot, device=a.device)[:num_mask]
            base_input = enc.get_base_input()
            base_input_override = _apply_spot_feature_mask(
                base_input=base_input,
                num_spot=N_spot,
                mask_idx=mask_idx,
                mode=str(getattr(a, "feat_mask_mode", "zero")),
                noise_std=float(getattr(a, "feat_noise_std", 0.02)),
            )

        Hs = enc(Ptilde, base_input_override=base_input_override)
        h_all = torch.cat(Hs, dim=1)  # used for negative mining scores

        # negative samples (Scheme B)
        neg_mult = 2

        use_layer_neg = bool(a.layer_aware) and (layer_ids is not None)

        ni, nj = sample_negatives(
            P,
            num_neg=neg_mult * pos.size(0),
            h=h_all,
            layer_ids=(layer_ids if use_layer_neg else None),
            layer_margin=(int(a.neg_layer_margin) if (use_layer_neg and hasattr(a, "neg_layer_margin")) else 2),
            layer_gamma=(float(a.layer_gamma) if use_layer_neg else 0.0),
            neg_adj_ratio=(float(getattr(a, "neg_adj_ratio", 0.3)) if use_layer_neg else 0.0),
            hard_ratio=float(a.neg_hard_ratio) if hasattr(a, "neg_hard_ratio") else 0.7,
            oversample_factor=int(a.neg_oversample) if hasattr(a, "neg_oversample") else 6,
            spatial=spatial_np,
            normals=normals,
            normal_scales=normal_scales,
            normal_margin=float(getattr(a, "normal_margin", 1.0)),
            normal_gamma=float(getattr(a, "normal_gamma", 2.0)),
        )

        ni = ni.to(a.device)
        nj = nj.to(a.device)

        # scores
        s_pos = scorer.edge_score(Hs, pos[:, 0], pos[:, 1])
        s_neg = scorer.edge_score(Hs, ni, nj)

        y_pos = torch.ones_like(s_pos, dtype=torch.float32, device=a.device)
        y_neg = torch.zeros_like(s_neg, dtype=torch.float32, device=a.device)
        y = torch.cat([y_pos, y_neg], dim=0)
        s = torch.cat([s_pos, s_neg], dim=0)
        s = torch.nan_to_num(
            s, nan=0.0, posinf=30.0, neginf=-30.0
        ).clamp_(-30.0, 30.0)
        loss = bce(s, y)

        loss_recon = None
        if (feat_decoder is not None) and (recon_crit is not None) and (mask_idx is not None) and (mask_idx.numel() > 0):
            h_all = node_embedding_from_layers(Hs, mode=a.embed_agg)  # [N_all, dec_in_dim]
            h_spot = h_all[:N_spot]

            h_mask = h_spot[mask_idx]  # [num_mask, dec_in_dim]
            x_target = X_recon[mask_idx]  # [num_mask, feat_dim]

            x_pred = feat_decoder(h_mask)
            loss_recon = recon_crit(x_pred, x_target)

            loss = loss + a.lambda_recon * loss_recon

        # backward and update
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        if a.scheduler:
            scheduler.step()

        if (ep + 1) % max(1, a.epochs // 10) == 0:
            with torch.inference_mode():
                logit_abs = s.abs().mean().item()
                p_pos = s_pos.sigmoid().mean().item()
                p_neg = s_neg.sigmoid().mean().item()
                if loss_recon is not None:
                    recon_val = loss_recon.item()
                else:
                    recon_val = None
            msg = (
                f"[epoch {ep + 1}/{a.epochs}] loss={loss.item():.4f} "
                f"| mean|logit|={logit_abs:.2f} | p_pos={p_pos:.3f} p_neg={p_neg:.3f} "
                f"| n_pos={int(s_pos.numel())} n_neg={int(s_neg.numel())}"
            )
            if recon_val is not None:
                msg += f" | loss_recon={recon_val:.4f}"
            print(msg)

    # ---------- extract embedding ----------
    enc.eval()
    with torch.inference_mode():
        Hs = enc(Ptilde)
        Z_all = node_embedding_from_layers(Hs, mode=a.embed_agg).cpu().numpy()

    N_spot = A.shape[0]
    Z_spot = Z_all[:N_spot]
    adata.obsm["emb"] = np.asarray(Z_spot)

    adata.obsp["A_spatial"] = A
    rep_key = a.use_rep.strip()

    if not rep_key:
        for k in ["X_dom", "X_emb", "X_pca", "X_umap", "X"]:
            if k in adata.obsm:
                rep_key = k
                break

    Z_spot = np.asarray(Z_spot, dtype=np.float32)
    if Z_spot.ndim != 2:
        raise ValueError(
            f"Z_spot dimension error: expected 2 dimensions, but got {Z_spot.ndim} dimensions"
        )

    if not rep_key:
        rep_key = "X_emb"

    adata.obsm[rep_key] = Z_spot
    adata.obsm["emb"] = Z_spot
    print(f"Saved embedding to adata.obsm['{rep_key}'] and adata.obsm['emb'].")

    # ---------- extra file for clustering ----------
    try:
        extra_path = f"{prefix}.augK{a.K}_d{a.dim}_for_cluster.npz"

        A_coo = A.tocoo()
        np.savez_compressed(
            extra_path,
            embedding=Z_spot.astype(np.float32),
            obs_names=np.array(adata.obs_names),
            spatial=(spatial_np if "spatial" in adata.obsm else None),
            A_row=A_coo.row.astype(np.int64),
            A_col=A_coo.col.astype(np.int64),
            A_weight=A_coo.data.astype(np.float32),
            rep_key=np.array([rep_key]),
            dim=np.array([a.dim], dtype=np.int64),
            K=np.array([a.K], dtype=np.int64),
        )
        print(f"Extra clustering file saved to: {extra_path}")
    except Exception as e:
        print(f"Error when saving extra clustering file: {e}")

    # ---------- skip clustering, only save embeddings ----------
    print("Clustering step has been removed: this script now ONLY learns embeddings.")

    adata.obsm[rep_key] = Z_spot
    adata.obsm["emb"] = Z_spot

    out_h5 = f"{prefix}.augK{a.K}_d{a.dim}_none.h5ad"
    print(f"[output] prefix = {prefix}")

    adata.write_h5ad(out_h5)
    print(f"Main output saved to: {out_h5}. (Only embeddings, no clustering labels)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--graph_model", choices=["Radius", "KNN"], default="KNN")
    p.add_argument("--radius", type=float, default=80.0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--topN", type=int, default=50)
    p.add_argument("--use_scanpy_workflow", action="store_true")
    p.add_argument("--pca_comps", type=int, default=50)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--pos_per_epoch", type=int, default=20000)
    p.add_argument("--embed_agg", choices=["concat", "mean", "last"], default="concat")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict_determinism", action="store_true", help="Enable strict deterministic algorithms. Slower and may require CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use_rep", type=str, default="")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu",)
    p.add_argument("--lambda_recon",type=float,default=0.1,help="weight for masked feature reconstruction loss (MAE-style)",)
    p.add_argument("--mask_ratio_feat",type=float,default=0.3,help="ratio of spots to mask for feature reconstruction (0~1)",)
    p.add_argument("--feat_mask_mode", type=str, default="zero", choices=["zero", "noise", "mean"], help="How to mask spot input features before encoder.")
    p.add_argument("--feat_noise_std", type=float, default=0.02, help="Std for Gaussian noise when feat_mask_mode='noise'.")
    p.add_argument("--recon_target", type=str, default="pca", choices=["pca"], help="Feature target used by reconstruction decoder.")
    p.add_argument("--activation",choices=["relu", "prelu", "gelu"],default="relu",help="activation function for encoder/scorer/decoder",)
    p.add_argument("--scheduler", action="store_true", help="whether to use learning rate scheduler (CosineAnnealingLR)")
    p.add_argument("--weight_decay", type=float, default=0.04, help="optimizer weight decay (e.g., for Adam)")
    p.add_argument('--out_prefix', type=str, default="", help='Output prefix for result files')
    p.add_argument("--layer_aware", action="store_true")
    p.add_argument("--layer_gamma", type=float, default=3.0)
    p.add_argument("--layer_axis", type=int, default=1)
    p.add_argument("--layer_scale_mode",type=str,default="edge_median",choices=["edge_median", "global_std"],help="How to scale dy in layer-aware edge reweighting.")
    p.add_argument("--layer_cutoff",type=float,default=None,help="Optional cutoff multiplier: drop edges with dy > layer_cutoff * scale.")
    p.add_argument("--neg_layer_margin", type=int, default=2,help="Minimum layer distance for negatives when layer_aware is enabled. 0 disables.")
    p.add_argument("--neg_hard_ratio", type=float, default=0.7,help="Hard negative ratio in Scheme B (top-k by similarity).")
    p.add_argument("--neg_oversample", type=int, default=6,help="Oversampling factor for candidate negatives before hard mining.")
    p.add_argument("--layer_key",type=str,default=None,help="Column name in adata.obs that stores layer labels (e.g. 'sce.layer_guess');If None, the code will try to infer automatically.")
    p.add_argument("--neg_adj_ratio", type=float, default=0.3, help="When layer_aware enabled, probability to sample negatives from adjacent layers (|Delta|=1). Suggest 0.25~0.45")
    p.add_argument("--pseudo_layer_bins", type=int, default=8,help="If no layer_key is found, derive pseudo-layer ids by binning spatial depth into this many bins (0 disables).")
    p.add_argument("--no_layer_fallback", action="store_true",help="If set, do NOT auto-detect sce.layer_guess etc. Keep self-supervised.")
    p.add_argument("--pseudo_layer_knn", type=int, default=18,help="k for spatial kNN graph used in pseudo-layer construction (only when pseudo_layer_bins>0).")
    p.add_argument("--pseudo_layer_binning", type=str, default="quantile", choices=["quantile", "uniform"],help="How to bin pseudo-depth into layers: quantile is more robust.")
    p.add_argument("--normal_aware", action="store_true",help="Use local normal-based cross-layer prior (self-supervised).")
    p.add_argument("--normal_knn", type=int, default=10, help="kNN size for local normal estimation.")
    p.add_argument("--normal_margin", type=float, default=1.0, help="Min normalized normal-distance for negatives.")
    p.add_argument("--normal_gamma", type=float, default=2.0,help="Exp weight on normalized normal-distance for negatives.")

    args = p.parse_args()
    main(args)
