from typing import Tuple

import torch
from torch import Tensor


def stratified_sample_balanced(x_cand: Tensor, y_cand: Tensor, n: int) -> Tuple[Tensor, Tensor]:
    target_pos = n // 2
    target_neg = n - target_pos

    pos_idx = torch.nonzero(y_cand > 0.5, as_tuple=False).squeeze(1)
    neg_idx = torch.nonzero(y_cand <= 0.5, as_tuple=False).squeeze(1)

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        raise ValueError("Candidate pool missing a class; increase n_cand or adjust seed.")

    pos_sel = pos_idx[torch.randint(0, pos_idx.numel(), (target_pos,))]
    neg_sel = neg_idx[torch.randint(0, neg_idx.numel(), (target_neg,))]

    x = torch.cat([x_cand[pos_sel], x_cand[neg_sel]], dim=0)
    y = torch.cat([torch.ones(target_pos), torch.zeros(target_neg)], dim=0)
    perm = torch.randperm(n)
    return x[perm], y[perm]

