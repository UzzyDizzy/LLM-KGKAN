#utils.py
import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mmd_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_tensor(0.0)

    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    z = torch.cat([x, y], dim=0)
    dist_sq = torch.cdist(z, z, p=2).pow(2)

    with torch.no_grad():
        positive = dist_sq[dist_sq > 0]
        bandwidth = positive.median().clamp_min(1e-6) if positive.numel() else dist_sq.new_tensor(1.0)

    kernels = dist_sq.new_zeros(dist_sq.shape)
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        gamma = 1.0 / (2.0 * bandwidth * scale)
        kernels = kernels + torch.exp(-gamma * dist_sq)

    nx = x.size(0)
    k_xx = kernels[:nx, :nx].mean()
    k_yy = kernels[nx:, nx:].mean()
    k_xy = kernels[:nx, nx:].mean()
    return k_xx + k_yy - 2.0 * k_xy


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    return ((preds == labels) & mask).sum().item() / mask.sum().item()


def build_datasets(cfg, kg, src_domain, tgt_domain, k_shot=None):
    from data import ABSADataset
    from sampling import few_shot_sample
    from domains import DOMAINS

    src_path = DOMAINS[src_domain]
    tgt_path = DOMAINS[tgt_domain]

    src = ABSADataset(src_path, cfg, kg, domain_id=0)
    tgt = ABSADataset(tgt_path, cfg, kg, domain_id=1)

    if k_shot is not None:
        tgt = few_shot_sample(tgt, k_shot)

    return src, tgt
