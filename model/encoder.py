from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import Optional, Tuple

def scipy_to_torch_coo(M: sp.coo_matrix) -> torch.Tensor:
    if not sp.isspmatrix_coo(M):
        M = M.tocoo()
    if M.dtype != np.float32:
        M = M.astype(np.float32)

    # Avoid torch.from_numpy() for environments with NumPy/PyTorch ABI mismatch.
    idx = torch.stack(
        [
            torch.tensor(M.row, dtype=torch.long),
            torch.tensor(M.col, dtype=torch.long),
        ],
        dim=0,
    )
    val = torch.tensor(np.asarray(M.data, dtype=np.float32), dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=M.shape, dtype=torch.float32).coalesce()

def normalize_rows_sparse_torch(M: torch.Tensor) -> torch.Tensor:
    assert M.is_sparse
    rows = M.indices()[0]
    vals = M.values()
    n = M.size(0)
    rowsum = torch.zeros(n, dtype=torch.float32, device=vals.device).index_add_(0, rows, vals)
    rowsum[rowsum == 0] = 1.0
    scale = 1.0 / rowsum
    vals = vals * scale[rows]
    return torch.sparse_coo_tensor(M.indices(), vals, M.size(), dtype=torch.float32, device=vals.device).coalesce()


def _contains_linear_edges(linear_idx: torch.Tensor, pos_sorted: torch.Tensor) -> torch.Tensor:
    if linear_idx.numel() == 0:
        return torch.zeros_like(linear_idx, dtype=torch.bool)
    hit = torch.searchsorted(pos_sorted, linear_idx)
    in_range = hit < pos_sorted.numel()
    matched = torch.zeros_like(in_range, dtype=torch.bool)
    if in_range.any():
        hit_ok = hit[in_range]
        matched[in_range] = pos_sorted[hit_ok] == linear_idx[in_range]
    return matched


def _pairwise_dot_scores(
    h: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    chunk_size: int = 131072,
) -> torch.Tensor:
    n = i_idx.numel()
    if n <= chunk_size:
        return (h[i_idx] * h[j_idx]).sum(dim=1)

    scores = []
    for st in range(0, n, chunk_size):
        ed = min(st + chunk_size, n)
        scores.append((h[i_idx[st:ed]] * h[j_idx[st:ed]]).sum(dim=1))
    return torch.cat(scores, dim=0)

class CollaborativeEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        dim: int = 64,
        K: int = 2,
        init_embed: torch.Tensor | None = None,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.K = K
        self.activation = activation

        # layers expect input feature dim == `dim`
        self.lin0 = nn.Linear(dim, dim)
        self.bn0 = nn.BatchNorm1d(dim)
        self.drop0 = nn.Dropout(0.2)
        self.lins = nn.ModuleList([nn.Linear(dim, dim) for _ in range(K)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(K)])
        self.drops = nn.ModuleList([nn.Dropout(0.2) for _ in range(K)])

        self.proj: nn.Linear | None = None
        self.embed_raw: nn.Parameter | None = None
        self.embed: nn.Parameter | None = None

        if init_embed is not None:
            assert init_embed.shape[0] == num_nodes
            in_dim = init_embed.shape[1]
            if in_dim == dim:
                # old behavior
                self.embed = nn.Parameter(init_embed)
            else:
                # learnable projection: in_dim -> dim
                self.embed_raw = nn.Parameter(init_embed)
                self.proj = nn.Linear(in_dim, dim)
        else:
            self.embed = nn.Parameter(torch.randn(num_nodes, dim) * 0.02)

    def get_base_input(self) -> torch.Tensor:
        if self.embed_raw is not None:
            return self.embed_raw
        return self.embed

    def get_base_embedding(self) -> torch.Tensor:
        base_input = self.get_base_input()
        if self.proj is not None:
            return self.proj(base_input)
        return base_input

    def forward(self, Ptilde: torch.Tensor, base_input_override: torch.Tensor | None = None):
        if base_input_override is not None:
            if self.proj is not None:
                base = self.proj(base_input_override)
            else:
                base = base_input_override
        else:
            base = self.get_base_embedding()

        Hk = self.lin0(base)
        Hk = self.bn0(Hk)
        Hk = self.activation(Hk)
        Hk = self.drop0(Hk)

        Hs = [Hk]
        for k in range(self.K):
            Hk = torch.sparse.mm(Ptilde, Hk)
            Hk = self.lins[k](Hk)
            Hk = self.bns[k](Hk)
            Hk = self.activation(Hk)
            Hk = self.drops[k](Hk)
            Hs.append(Hk)
        return Hs

class CrossLayerScorer(nn.Module):
    def __init__(self, dim: int = 64, K: int = 2, hidden: int = 128, activation: nn.Module = nn.ReLU()):
        super().__init__()
        in_dim = dim * (K + 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            activation,
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            activation,
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            activation,
            nn.Linear(hidden // 2, 1)
        )
    def edge_score(self, Hs, i_idx: torch.Tensor, j_idx: torch.Tensor):
        feats = []
        for H in Hs:
            feats.append(H[i_idx] * H[j_idx])   # Hadamard
        x = torch.cat(feats, dim=1)
        return self.mlp(x).squeeze(-1)


def sample_negatives(
    P,  # torch sparse COO adjacency (E_coo)
    num_neg: int,
    h: Optional[torch.Tensor] = None,
    encoder: Optional[nn.Module] = None,
    Ptilde: Optional[torch.Tensor] = None,
    layer_ids: Optional[torch.Tensor] = None,
    layer_margin: int = 0,
    layer_gamma: float = 0.0,
    neg_adj_ratio: float = 0.0,
    hard_ratio: float = 0.7,
    oversample_factor: int = 6,
    # ---------- normal-aware (self-supervised; uses only spatial) ----------
    spatial=None,              # (N,2) numpy or torch
    normals=None,              # (N,2) numpy or torch (unit vectors)
    normal_scales=None,        # (N,)  numpy or torch (robust scale per i)
    normal_margin: float = 0.0,  # min normalized normal-distance for negatives
    normal_gamma: float = 0.0,   # exp weight on normal-distance
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Negative sampling with optional:
      - layer-aware constraints (discrete layer_ids)
      - normal-aware constraints (continuous local normal distance; self-supervised)
      - Scheme-B hard mining (dot-product scoring) + random remainder

    P: sparse adjacency (torch.sparse_coo_tensor)
    h: node embedding used for hardness scoring (if None, will compute via encoder+Ptilde once)
    """

    # -------- basic checks --------
    assert P is not None and P.is_sparse, "P must be a torch sparse COO tensor."
    E_coo = P.coalesce()
    device = E_coo.device

    # ---- Degree-based sampling probability (favor low-degree nodes) ----
    ii = E_coo.indices()[0]
    jj = E_coo.indices()[1]
    n_nodes = int(E_coo.size(0))
    degrees = torch.bincount(ii, minlength=n_nodes) + torch.bincount(jj, minlength=n_nodes)
    prob = 1.0 / (degrees.float() + 1e-8)
    prob = prob / prob.sum()

    # ---- Existing edges as sorted linear index (vectorized membership test) ----
    pos_lin = torch.cat([ii * n_nodes + jj, jj * n_nodes + ii], dim=0).to(torch.long)
    pos_lin = torch.unique(pos_lin, sorted=True)

    # ---- Prepare embeddings for hardness scoring ----
    if h is None:
        assert encoder is not None and Ptilde is not None, "Need `h` or (encoder + Ptilde)."
        was_training = encoder.training
        encoder.eval()
        with torch.inference_mode():
            Hs = encoder(Ptilde)
            h = torch.cat(Hs, dim=1)
        encoder.train(was_training)

    assert h is not None
    # h should be on some device (likely cuda); we’ll sample indices on that device
    h_device = h.device

    # ---- Layer utilities (only for spot nodes) ----
    layer2idx = None
    layer_weight = None
    max_layer = None
    n_spot = None
    if layer_ids is not None:
        if layer_ids.device != device:
            layer_ids = layer_ids.to(device)
        n_spot = int(layer_ids.numel())
        max_layer = int(layer_ids.max().item()) if n_spot > 0 else None

        if max_layer is not None:
            layer2idx = [torch.where(layer_ids == lid)[0] for lid in range(max_layer + 1)]
            li_grid = torch.arange(max_layer + 1, device=device).unsqueeze(1)
            lj_grid = torch.arange(max_layer + 1, device=device).unsqueeze(0)
            dist = (li_grid - lj_grid).abs()

            layer_weight = torch.ones_like(dist, dtype=torch.float32)
            if int(layer_margin) > 0:
                layer_weight = torch.where(
                    dist >= int(layer_margin),
                    layer_weight,
                    torch.zeros_like(layer_weight),
                )
            if float(layer_gamma) > 0:
                layer_weight = torch.where(
                    layer_weight > 0,
                    torch.exp(float(layer_gamma) * dist.float()),
                    torch.zeros_like(layer_weight),
                )

            row_sum = layer_weight.sum(dim=1, keepdim=True)
            layer_weight = layer_weight / row_sum.clamp_min(1e-12)
            zero_rows = row_sum.squeeze(1) <= 0
            if zero_rows.any():
                layer_weight[zero_rows] = 1.0 / float(max_layer + 1)

    # ---------- normal-aware tensor prep (CPU is OK; we only need indexing) ----------
    normal_enabled = (normals is not None) and (spatial is not None) and ((normal_gamma > 0) or (normal_margin > 0))
    if normal_enabled:
        # move to CPU tensors for cheap indexing; compute d on CPU then ship weights to h_device
        if torch.is_tensor(spatial):
            xy = spatial.detach().to("cpu").float()
        else:
            xy = torch.tensor(np.asarray(spatial, dtype=np.float32), dtype=torch.float32, device="cpu")

        if torch.is_tensor(normals):
            nnv = normals.detach().to("cpu").float()
        else:
            nnv = torch.tensor(np.asarray(normals, dtype=np.float32), dtype=torch.float32, device="cpu")

        if normal_scales is None:
            sc = torch.ones(xy.size(0), dtype=torch.float32, device="cpu")
        else:
            if torch.is_tensor(normal_scales):
                sc = normal_scales.detach().to("cpu").float()
            else:
                sc = torch.tensor(np.asarray(normal_scales, dtype=np.float32), dtype=torch.float32, device="cpu")

        # sanity
        assert xy.ndim == 2 and xy.size(1) == 2
        assert nnv.ndim == 2 and nnv.size(1) == 2
        assert sc.ndim == 1 and sc.numel() == xy.size(0)

    # ---- Generate candidates (oversample) ----
    n_cand = max(int(num_neg * oversample_factor), num_neg + 100)

    cand_i_parts = []
    cand_j_parts = []

    max_attempts = max(20, oversample_factor * 30)
    attempts = 0
    collected = 0
    base_batch = int(min(262144, max(4096, n_cand)))

    while collected < n_cand and attempts < max_attempts:
        attempts += 1
        need = n_cand - collected
        batch = int(min(262144, max(base_batch, need * 2)))

        i = torch.multinomial(prob, num_samples=batch, replacement=True)
        j = torch.multinomial(prob, num_samples=batch, replacement=True)
        is_adj = torch.zeros(batch, dtype=torch.bool, device=device)

        if layer_ids is not None and n_spot is not None and n_spot > 0 and layer_weight is not None:
            spot_mask = i < n_spot
            if spot_mask.any():
                i_spot = i[spot_mask]
                li_spot = layer_ids[i_spot]

                lj_spot = torch.multinomial(layer_weight[li_spot], num_samples=1).squeeze(1)
                is_adj_spot = torch.zeros(li_spot.numel(), dtype=torch.bool, device=device)

                if (neg_adj_ratio is not None) and (float(neg_adj_ratio) > 0) and (max_layer is not None) and (max_layer >= 1):
                    use_adj = torch.rand(li_spot.numel(), device=device) < float(neg_adj_ratio)
                    left_ok = li_spot > 0
                    right_ok = li_spot < max_layer
                    can_adj = use_adj & (left_ok | right_ok)

                    if can_adj.any():
                        li_adj = li_spot[can_adj]
                        left_adj = li_adj > 0
                        right_adj = li_adj < max_layer
                        choose_right = torch.rand(li_adj.numel(), device=device) < 0.5
                        lj_adj = torch.where(
                            left_adj & right_adj,
                            torch.where(choose_right, li_adj + 1, li_adj - 1),
                            torch.where(left_adj, li_adj - 1, li_adj + 1),
                        )
                        lj_spot[can_adj] = lj_adj
                        is_adj_spot[can_adj] = True

                j_spot = torch.empty_like(i_spot)
                for lid in torch.unique(lj_spot).tolist():
                    lid_int = int(lid)
                    lid_mask = lj_spot == lid_int
                    cnt = int(lid_mask.sum().item())
                    pool = layer2idx[lid_int]
                    if pool.numel() > 0:
                        ridx = torch.randint(0, pool.numel(), (cnt,), device=device)
                        j_spot[lid_mask] = pool[ridx]
                    else:
                        j_spot[lid_mask] = torch.randint(0, n_spot, (cnt,), device=device)
                        is_adj_spot[lid_mask] = False

                j[spot_mask] = j_spot
                is_adj[spot_mask] = is_adj_spot

        valid = i != j
        lin = i * n_nodes + j
        valid &= ~_contains_linear_edges(lin, pos_lin)

        if layer_ids is not None and n_spot is not None and int(layer_margin) > 0:
            spot_pair = (i < n_spot) & (j < n_spot)
            enforce = spot_pair & (~is_adj)
            if enforce.any():
                li_enf = layer_ids[i[enforce]]
                lj_enf = layer_ids[j[enforce]]
                margin_ok = (li_enf - lj_enf).abs() >= int(layer_margin)
                enforce_idx = torch.where(enforce)[0]
                valid[enforce_idx] &= margin_ok

        if valid.any():
            vi = i[valid]
            vj = j[valid]
            take = min(need, int(vi.numel()))
            cand_i_parts.append(vi[:take].to(h_device))
            cand_j_parts.append(vj[:take].to(h_device))
            collected += take

    if collected < num_neg:
        raise RuntimeError(
            f"Not enough negative candidates ({collected}) for num_neg={num_neg}. "
            f"Try reducing layer_margin/layer_gamma or increasing oversample_factor."
        )

    # candidate index tensors (on h_device for scoring)
    cand_i_t = torch.cat(cand_i_parts, dim=0)
    cand_j_t = torch.cat(cand_j_parts, dim=0)
    raw_cand_i_t = cand_i_t
    raw_cand_j_t = cand_j_t

    # ---- Hardness score: dot product ----
    with torch.inference_mode():
        scores = _pairwise_dot_scores(h, cand_i_t, cand_j_t)  # (n_cand,)

    # ---- normal-aware: filter and/or reweight candidate scores ----
    if normal_enabled:
        ci_cpu = cand_i_t.detach().to("cpu")
        cj_cpu = cand_j_t.detach().to("cpu")

        n_spatial = xy.size(0)  # = N_spot (e.g. 3460)

        # only apply normal-aware to spot-spot pairs
        mask = (ci_cpu < n_spatial) & (cj_cpu < n_spatial)

        # default d = 0 for non-spot pairs (no effect)
        d = torch.zeros(ci_cpu.size(0), dtype=torch.float32, device="cpu")

        if mask.any():
            ci_m = ci_cpu[mask]
            cj_m = cj_cpu[mask]

            dxy = xy[cj_m] - xy[ci_m]  # (M,2)
            proj = (dxy * nnv[ci_m]).sum(dim=1).abs()  # (M,)
            d_m = proj / (sc[ci_m] + 1e-6)  # normalized distance
            d[mask] = d_m

        # margin filter: only filter spot-spot pairs, keep all non-spot pairs
        if normal_margin > 0 and mask.any():
            keep = (~mask) | (d >= float(normal_margin))

            # if too aggressive, relax once
            if keep.sum().item() >= max(num_neg + 100, int(1.2 * num_neg)):
                keep_gpu = keep.to(h_device)
                cand_i_t = cand_i_t[keep_gpu]
                cand_j_t = cand_j_t[keep_gpu]
                scores = scores[keep_gpu]
                d = d[keep]
                mask = mask[keep]  # keep mask in sync
            else:
                # not enough kept; skip hard filtering this round
                pass

        # exp reweight: apply ONLY to spot-spot pairs; others weight=1
        if normal_gamma > 0:
            w = torch.ones_like(d, dtype=torch.float32, device="cpu")
            if mask.any():
                w[mask] = torch.exp(float(normal_gamma) * d[mask]).clamp(max=1e4)
            scores = scores * w.to(h_device)

    # ---- Select final negatives: hard + random ----
    k_hard = int(num_neg * float(hard_ratio))
    k_hard = max(0, min(k_hard, num_neg))

    # if after filtering we have fewer candidates than needed, fallback (should be rare)
    if scores.numel() < num_neg:
        # fallback: ignore normal filter and just random sample from original cand lists
        cand_i_t = raw_cand_i_t
        cand_j_t = raw_cand_j_t
        with torch.inference_mode():
            scores = _pairwise_dot_scores(h, cand_i_t, cand_j_t)
        if scores.numel() < num_neg:
            raise RuntimeError(f"Too few candidates after filtering: {scores.numel()} < num_neg={num_neg}")

    hard_idx = torch.topk(scores, k=k_hard, largest=True).indices  # tensor on h_device
    chosen_i = cand_i_t[hard_idx]
    chosen_j = cand_j_t[hard_idx]

    remaining = num_neg - chosen_i.numel()
    if remaining > 0:
        hard_mask = torch.zeros(scores.numel(), dtype=torch.bool, device=h_device)
        hard_mask[hard_idx] = True
        pool = torch.where(~hard_mask)[0]
        if pool.numel() == 0:
            rand_idx = torch.randint(0, scores.numel(), (remaining,), device=h_device)
        elif pool.numel() < remaining:
            # if pool too small, sample with replacement
            rand_idx = pool[torch.randint(0, max(1, pool.numel()), (remaining,), device=h_device)]
        else:
            rand_idx = pool[torch.randperm(pool.numel(), device=h_device)[:remaining]]

        chosen_i = torch.cat([chosen_i, cand_i_t[rand_idx]], dim=0)
        chosen_j = torch.cat([chosen_j, cand_j_t[rand_idx]], dim=0)

    # return on the SAME device as E_coo (consistent with your original code)
    ni = chosen_i.to(device)
    nj = chosen_j.to(device)
    return ni, nj


class FeatureDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            activation,
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            activation,
            nn.Linear(hidden, out_dim)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)

@torch.no_grad()
def node_embedding_from_layers(Hs, mode: str = "concat"):
    if mode == "concat":
        return torch.cat(Hs, dim=1)
    elif mode == "mean":
        return torch.stack(Hs, dim=0).mean(dim=0)
    else:
        return Hs[-1]
