from typing import List, Optional
import numpy as np
import torch
import pynvml
from sklearn.utils.extmath import randomized_svd
from torch import Tensor

def get_free_gpu() -> int:

    index = 0
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        max = 0
        for i in range(torch.cuda.device_count()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            index = i if info.free > max else index
            max = info.free if info.free > max else max
    return index

def dual_pca(
    X: np.ndarray,
    Y: np.ndarray,
    dim: Optional[int] = 50,
    singular: Optional[bool] = True,
    backend: Optional[str] = "sklearn",
    use_gpu: Optional[bool] = True,
) -> List[Tensor]:

    assert X.shape[1] == Y.shape[1]
    device = torch.device(
        f"cuda:{get_free_gpu()}" if torch.cuda.is_available() and use_gpu else "cpu"
    )
    X = torch.Tensor(X).to(device=device)
    Y = torch.Tensor(Y).to(device=device)
    cor_var = X @ Y.T
    if backend == "torch":
        U, S, Vh = torch.linalg.svd(cor_var)
        if not singular:
            return U[:, :dim], Vh.T[:, :dim]
        Z_x = U[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        Z_y = Vh.T[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        return Z_x.cpu(), Z_y.cpu()

    elif backend == "sklearn":
        cor_var = cor_var.cpu().numpy()
        U, S, Vh = randomized_svd(cor_var, n_components=dim, random_state=0)
        if not singular:
            return Tensor(U), Tensor(Vh.T)
        Z_x = U @ np.sqrt(np.diag(S))
        Z_y = Vh.T @ np.sqrt(np.diag(S))
        return Tensor(Z_x), Tensor(Z_y)
