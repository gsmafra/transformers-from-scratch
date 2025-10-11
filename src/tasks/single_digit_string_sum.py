from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .base import Task


# Fixed vocabulary for single-digit equations like "1+2=3"
_TOKENS: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "="]
_TOK2IDX: Dict[str, int] = {ch: i for i, ch in enumerate(_TOKENS)}


def _one_hot(indices: Tensor, vocab_size: int) -> Tensor:
    out = torch.zeros(indices.size(0), indices.size(1), vocab_size, dtype=torch.float32, device=indices.device)
    out.scatter_(2, indices.unsqueeze(-1).long(), 1.0)
    return out


class SingleDigitStringSumTask(Task):

    feature_dim: int = len(_TOKENS)

    def label(self, x: Tensor) -> Tensor:
        """Label is 1.0 if the equation "a+b=c" is correct, else 0.0.

        Expects one-hot tokens over the fixed vocabulary of size 12 and
        sequence length T == 5 corresponding to tokens: d + d = d.
        """
        if x.dim() != 3 or x.size(1) != 5 or x.size(2) < len(_TOKENS):
            raise ValueError("SingleDigitStringSumTask expects shape (N, 5, V>=12) with one-hot tokens")

        idx = x.argmax(dim=2)  # (N, 5)
        a = idx[:, 0]
        plus = idx[:, 1]
        b = idx[:, 2]
        eq = idx[:, 3]
        c = idx[:, 4]

        # Validate operators
        plus_ok = plus.eq(_TOK2IDX["+"])
        eq_ok = eq.eq(_TOK2IDX["="])

        # Map indices for digits into numeric values (assume 0..9 map directly)
        a_val = a.clamp_max(9).to(torch.long)
        b_val = b.clamp_max(9).to(torch.long)
        c_val = c.clamp_max(9).to(torch.long)

        correct_sum = a_val + b_val == c_val
        ok = plus_ok & eq_ok & correct_sum
        return ok.float()

    def generate_candidates(self, n: int, T: int) -> Tuple[Tensor, Tensor]:
        """Generate valid equations "a+b=c" with both correct and incorrect labels.

        Returns one-hot tokens of shape (n, 5, F). Requires F >= 12 and T == 5.
        Half the samples are correct (y=1), half incorrect (y=0), shuffled.
        """
        if T != 5:
            raise ValueError("SingleDigitStringSumTask requires sequence_length T == 5 (format d+d=d)")
        F = self.feature_dim

        target_pos = n // 2
        target_neg = n - target_pos

        # Positives: choose (a,b) such that a+b <= 9, set c = a+b
        pos_pairs: List[Tuple[int, int]] = [(a, b) for a in range(10) for b in range(10) if a + b <= 9]
        pos_choice = torch.randint(0, len(pos_pairs), (target_pos,))
        a_pos = torch.tensor([pos_pairs[i][0] for i in pos_choice.tolist()], dtype=torch.long)
        b_pos = torch.tensor([pos_pairs[i][1] for i in pos_choice.tolist()], dtype=torch.long)
        c_pos = a_pos + b_pos

        pos_seq = torch.stack(
            [
                a_pos,
                torch.full_like(a_pos, _TOK2IDX["+"] ),
                b_pos,
                torch.full_like(a_pos, _TOK2IDX["="] ),
                c_pos,
            ],
            dim=1,
        )

        # Negatives: arbitrary (a,b); choose c != a+b
        a_neg = torch.randint(0, 10, (target_neg,))
        b_neg = torch.randint(0, 10, (target_neg,))
        c_true = a_neg + b_neg
        # pick a different single-digit result
        c_neg = (c_true + torch.randint(1, 10, (target_neg,))) % 10

        neg_seq = torch.stack(
            [
                a_neg,
                torch.full_like(a_neg, _TOK2IDX["+"] ),
                b_neg,
                torch.full_like(a_neg, _TOK2IDX["="] ),
                c_neg,
            ],
            dim=1,
        )

        seq_idx = torch.cat([pos_seq, neg_seq], dim=0)
        y = torch.cat([torch.ones(target_pos), torch.zeros(target_neg)], dim=0)

        # Shuffle
        perm = torch.randperm(n)
        seq_idx = seq_idx[perm]
        y = y[perm]

        x = _one_hot(seq_idx, F)
        return x, y
