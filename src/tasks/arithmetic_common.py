from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor

# Shared vocabulary for simple arithmetic strings over single digits
TOKENS: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "="]
TOK2IDX: Dict[str, int] = {ch: i for i, ch in enumerate(TOKENS)}


def one_hot(indices: Tensor, vocab_size: int) -> Tensor:
    out = torch.zeros(indices.size(0), indices.size(1), vocab_size, dtype=torch.float32)
    out.scatter_(2, indices.unsqueeze(-1).long(), 1.0)
    return out


# Allowed operator-form patterns expressed as (op1_idx, op2_idx)
FORM_A_PLUS_B_EQ_C: Tuple[int, int] = (TOK2IDX["+"], TOK2IDX["="])
FORM_A_EQ_B_PLUS_C: Tuple[int, int] = (TOK2IDX["="], TOK2IDX["+"])


def label_equations_from_indices(idx: Tensor, allowed_forms: Sequence[Tuple[int, int]]) -> Tensor:
    """Return float labels for indices of shape (N, 5) given allowed operator forms.

    Supports two forms:
      - a + b = c
      - a = b + c
    """
    a = idx[:, 0]
    op1 = idx[:, 1]
    b = idx[:, 2]
    op2 = idx[:, 3]
    c = idx[:, 4]

    a_val = a.clamp_max(9).to(torch.long)
    b_val = b.clamp_max(9).to(torch.long)
    c_val = c.clamp_max(9).to(torch.long)

    out = torch.zeros(idx.size(0), dtype=torch.float32)
    for f in allowed_forms:
        if f == FORM_A_PLUS_B_EQ_C:
            form_mask = op1.eq(TOK2IDX["+"]) & op2.eq(TOK2IDX["="])
            correct = (a_val + b_val) == c_val
            out = torch.maximum(out, (form_mask & correct).float())
        elif f == FORM_A_EQ_B_PLUS_C:
            form_mask = op1.eq(TOK2IDX["="]) & op2.eq(TOK2IDX["+"])
            correct = a_val == (b_val + c_val)
            out = torch.maximum(out, (form_mask & correct).float())
        else:
            # Unknown form: ignore
            continue
    return out


def generate_candidates_equations(n: int, T: int, allowed_forms: Sequence[Tuple[int, int]]) -> Tuple[Tensor, Tensor]:
    """Unified candidate generator for both operator orders.

    Returns balanced positives/negatives, one-hot encoded as (N, 5, V).
    """
    if T != 5:
        raise ValueError("Requires sequence_length T == 5 (formats d+d=d or d=d+d)")

    pos_list: List[Tensor] = []
    neg_list: List[Tensor] = []
    needed = (n + 1) // 2

    while len(pos_list) < needed:
        chunk = max(needed - len(pos_list), 128)

        if FORM_A_PLUS_B_EQ_C in allowed_forms:
            a = torch.randint(0, 10, (chunk,))
            b = torch.randint(0, 10, (chunk,))
            mask1 = (a + b) <= 9
            valid1 = torch.nonzero(mask1, as_tuple=False).squeeze(1)
            for i in valid1.tolist():
                av = int(a[i].item())
                bv = int(b[i].item())
                cv = av + bv
                pos_seq = torch.tensor([av, TOK2IDX["+"], bv, TOK2IDX["="], cv], dtype=torch.long)
                # Corrupt c
                offset = int(torch.randint(1, 10, ()).item())
                c_neg = (cv + offset) % 10
                neg_seq = torch.tensor([av, TOK2IDX["+"], bv, TOK2IDX["="], c_neg], dtype=torch.long)
                pos_list.append(pos_seq)
                neg_list.append(neg_seq)
                if len(pos_list) >= needed:
                    break

        if len(pos_list) >= needed:
            break

        if FORM_A_EQ_B_PLUS_C in allowed_forms:
            b2 = torch.randint(0, 10, (chunk,))
            c2 = torch.randint(0, 10, (chunk,))
            mask2 = (b2 + c2) <= 9
            valid2 = torch.nonzero(mask2, as_tuple=False).squeeze(1)
            for i in valid2.tolist():
                bv = int(b2[i].item())
                cv = int(c2[i].item())
                av = bv + cv
                pos_seq = torch.tensor([av, TOK2IDX["="], bv, TOK2IDX["+"], cv], dtype=torch.long)
                # Corrupt a
                offset = int(torch.randint(1, 10, ()).item())
                a_neg = (av + offset) % 10
                neg_seq = torch.tensor([a_neg, TOK2IDX["="], bv, TOK2IDX["+"], cv], dtype=torch.long)
                pos_list.append(pos_seq)
                neg_list.append(neg_seq)
                if len(pos_list) >= needed:
                    break

    seq_idx = torch.stack(pos_list + neg_list, dim=0)
    y = torch.cat([torch.ones(len(pos_list)), torch.zeros(len(neg_list))], dim=0)

    # Shuffle and truncate to n
    perm = torch.randperm(seq_idx.size(0))
    seq_idx = seq_idx[perm][:n]
    y = y[perm][:n]

    x = one_hot(seq_idx, len(TOKENS))
    return x, y
