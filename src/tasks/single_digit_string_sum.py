from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .base import Task


# Vocabulary for equations like "a+b=c" with single digits
_TOKENS: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "="]
_TOK2IDX: Dict[str, int] = {ch: i for i, ch in enumerate(_TOKENS)}


def _one_hot(indices: Tensor, vocab_size: int) -> Tensor:
    out = torch.zeros(indices.size(0), indices.size(1), vocab_size, dtype=torch.float32, device=indices.device)
    out.scatter_(2, indices.unsqueeze(-1).long(), 1.0)
    return out


class SingleDigitStringSumTask(Task):
    feature_dim: int = len(_TOKENS)

    def label(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(1) != 5 or x.size(2) < self.feature_dim:
            raise ValueError("Expected shape (N, 5, V>=12) with one-hot tokens")

        idx = x.argmax(dim=2)  # (N, 5)
        a = idx[:, 0]
        plus = idx[:, 1]
        b = idx[:, 2]
        eq = idx[:, 3]
        c = idx[:, 4]

        plus_ok = plus.eq(_TOK2IDX["+"])
        eq_ok = eq.eq(_TOK2IDX["="])

        a_val = a.clamp_max(9).to(torch.long)
        b_val = b.clamp_max(9).to(torch.long)
        c_val = c.clamp_max(9).to(torch.long)

        correct_sum = a_val + b_val == c_val
        ok = plus_ok & eq_ok & correct_sum
        return ok.float()

    def generate_candidates(self, n: int, T: int) -> Tuple[Tensor, Tensor]:
        """Generate pairs: for each random (a,b) with a+b <= 9, emit one correct and one incorrect sample.

        If a+b > 9, discard that pair (emit nothing). Truncate to n after shuffling.
        Output one-hot tokens with shape (N, 5, self.feature_dim).
        """
        if T != 5:
            raise ValueError("Requires sequence_length T == 5 (format d+d=d)")

        needed_pairs = (n + 1) // 2  # two samples (pos+neg) per valid pair
        pairs: List[Tuple[int, int]] = []

        # Keep sampling until we collect enough valid pairs
        while len(pairs) < needed_pairs:
            # Sample in chunks for efficiency
            chunk = max(needed_pairs - len(pairs), 128)
            a = torch.randint(0, 10, (chunk,))
            b = torch.randint(0, 10, (chunk,))
            mask = (a + b) <= 9
            valid = torch.nonzero(mask, as_tuple=False).squeeze(1)
            for i in valid.tolist():
                pairs.append((int(a[i].item()), int(b[i].item())))
                if len(pairs) >= needed_pairs:
                    break

        # Build positives and matched negatives per pair
        pos_list: List[Tensor] = []
        neg_list: List[Tensor] = []
        for a, b in pairs:
            c_true = a + b
            pos_seq = torch.tensor([a, _TOK2IDX["+"], b, _TOK2IDX["="], c_true], dtype=torch.long)

            # choose c_neg != c_true
            # pick an offset in 1..9 and wrap within 0..9
            offset = int(torch.randint(1, 10, ()).item())
            c_neg = (c_true + offset) % 10
            neg_seq = torch.tensor([a, _TOK2IDX["+"], b, _TOK2IDX["="], c_neg], dtype=torch.long)

            pos_list.append(pos_seq)
            neg_list.append(neg_seq)

        seq_idx = torch.stack(pos_list + neg_list, dim=0)
        y = torch.cat([torch.ones(len(pos_list)), torch.zeros(len(neg_list))], dim=0)

        # Shuffle and truncate to n
        perm = torch.randperm(seq_idx.size(0))
        seq_idx = seq_idx[perm][:n]
        y = y[perm][:n]

        x = _one_hot(seq_idx, self.feature_dim)
        return x, y

