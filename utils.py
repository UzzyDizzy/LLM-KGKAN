#utils.py
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist = x_norm - 2.0 * x @ y.t() + y_norm.t()
    return torch.exp(-dist / (2.0 * sigma ** 2))


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_tensor(0.0)
    k_xx = gaussian_kernel(x, x, sigma).mean()
    k_yy = gaussian_kernel(y, y, sigma).mean()
    k_xy = gaussian_kernel(x, y, sigma).mean()
    return k_xx + k_yy - 2.0 * k_xy


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    return ((preds == labels) & mask).sum().item() / mask.sum().item()


from data import ABSADataset
from sampling import few_shot_sample
from domains import DOMAINS


def build_datasets(cfg, kg, src_domain, tgt_domain, k_shot=None):
    src_path = DOMAINS[src_domain]
    tgt_path = DOMAINS[tgt_domain]

    src = ABSADataset(src_path, cfg, kg, domain_id=0)
    tgt = ABSADataset(tgt_path, cfg, kg, domain_id=1)

    if k_shot is not None:
        tgt = few_shot_sample(tgt, k_shot)

    return src, tgt
