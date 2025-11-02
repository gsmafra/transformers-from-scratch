from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from .arithmetic_common import TOKENS as _TOKENS, TOK2IDX as _TOK2IDX
from .base import Task


def _int_to_digits(n: int, D: int) -> List[int]:
    s = f"{n:0{D}d}"[-D:]
    return [int(ch) for ch in s]


class MultiDigitSumTask(Task):
    """Sum of multiple D-digit operands on each side: LHS == RHS.

    - Vocabulary: digits 0-9, plus '+', equals '=' (shared with other arithmetic tasks).
    - Format: `<D-digit> (+ <D-digit>)*N_left_minus1 = <D-digit> (+ <D-digit>)*N_right_minus1`.
    - Each operand is exactly D digits (zero-padded on the left if needed).
    - The number of operands per side is fixed at `N` for both sides (use N_left/N_right to differ).
    """

    feature_dim: int = len(_TOKENS)

    def __init__(self, D: int = 2, N: int = 2, *, N_left: int | None = None, N_right: int | None = None) -> None:
        super().__init__()
        self.D = int(D)
        self.N_left = int(N_left if N_left is not None else N)
        self.N_right = int(N_right if N_right is not None else N)
        if self.D <= 0 or self.N_left <= 0 or self.N_right <= 0:
            raise ValueError("D, N_left, N_right must be positive integers")

        # Precompute expected sequence length for shape checks
        # Total tokens: D*N_left + (N_left-1) '+' + 1 '=' + D*N_right + (N_right-1) '+'
        self.sequence_length = self.D * self.N_left + (self.N_left - 1) + 1 + self.D * self.N_right + (self.N_right - 1)

    def _parse_side(self, idx: Tensor, start: int, n_ops: int) -> Tuple[int, int]:
        """Parse a side starting at `start` with `n_ops` operands.

        Returns (sum_value, next_pos). Raises ValueError if format tokens mismatch.
        """
        pos = start
        base = 10 ** self.D
        total = 0
        for k in range(n_ops):
            # D digits
            digits = []
            for _ in range(self.D):
                token = int(idx[pos].item())
                if token > 9:
                    raise ValueError("Expected digit token 0-9 in operand")
                digits.append(token)
                pos += 1
            # Accumulate integer value
            val = 0
            for d in digits:
                val = val * 10 + d
            if not (0 <= val < base):
                raise ValueError("Parsed value out of range for D digits")
            total += val
            # '+' between operands, except after last
            if k < n_ops - 1:
                if int(idx[pos].item()) != _TOK2IDX["+"]:
                    raise ValueError("Expected '+' between operands")
                pos += 1
        return total, pos

    def label(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(2) < self.feature_dim:
            raise ValueError("Expected shape (N, T, V>=12) with one-hot tokens")
        if x.size(1) != self.sequence_length:
            raise ValueError(f"Expected sequence length T == {self.sequence_length}")

        idx = x.argmax(dim=2)  # (N, T)
        N = idx.size(0)
        out = torch.zeros(N, dtype=torch.float32)

        for i in range(N):
            row = idx[i]
            try:
                lhs_sum, pos = self._parse_side(row, 0, self.N_left)
                # '='
                if int(row[pos].item()) != _TOK2IDX["="]:
                    continue
                pos += 1
                rhs_sum, pos2 = self._parse_side(row, pos, self.N_right)
                if pos2 != self.sequence_length:
                    continue
            except ValueError:
                # Malformed format; label remains 0
                continue
            out[i] = 1.0 if lhs_sum == rhs_sum else 0.0
        return out

    def _encode_example(self, left: Sequence[int], right: Sequence[int]) -> Tensor:
        # Build index sequence for one example according to the fixed format
        parts: List[int] = []
        for j, v in enumerate(left):
            parts.extend(_int_to_digits(v, self.D))
            if j < len(left) - 1:
                parts.append(_TOK2IDX["+"])
        parts.append(_TOK2IDX["="])
        for j, v in enumerate(right):
            parts.extend(_int_to_digits(v, self.D))
            if j < len(right) - 1:
                parts.append(_TOK2IDX["+"])
        return torch.tensor(parts, dtype=torch.long)

    def generate_candidates(self, n: int, T: int) -> Tuple[Tensor, Tensor]:
        if T != self.sequence_length:
            raise ValueError(f"Requires sequence_length T == {self.sequence_length}")

        base = 10 ** self.D
        pos_list: List[Tensor] = []
        neg_list: List[Tensor] = []
        needed = (n + 1) // 2

        while len(pos_list) < needed:
            # Sample random operands for all but the final RHS operand, then solve for the last one
            left = torch.randint(0, base, (self.N_left,), dtype=torch.long)
            right_prefix = torch.randint(0, base, (max(self.N_right - 1, 0),), dtype=torch.long)
            lhs_sum = int(left.sum().item())
            rhs_partial = int(right_prefix.sum().item())
            rhs_last = lhs_sum - rhs_partial
            if 0 <= rhs_last < base:
                right = torch.cat([right_prefix, torch.tensor([rhs_last], dtype=torch.long)]) if self.N_right > 0 else right_prefix
                pos_seq = self._encode_example(left.tolist(), right.tolist())
                # Negative by perturbing last RHS operand
                offset = int(torch.randint(1, base, ()).item())
                rhs_last_neg = (rhs_last + offset) % base
                right_neg = right.clone()
                if self.N_right > 0:
                    right_neg[-1] = rhs_last_neg
                neg_seq = self._encode_example(left.tolist(), right_neg.tolist())
                pos_list.append(pos_seq)
                neg_list.append(neg_seq)

        seq_idx = torch.stack(pos_list + neg_list, dim=0)
        y = torch.cat([torch.ones(len(pos_list)), torch.zeros(len(neg_list))], dim=0)

        # Shuffle and truncate to n
        perm = torch.randperm(seq_idx.size(0))
        seq_idx = seq_idx[perm][:n]
        y = y[perm][:n]

        # One-hot
        x = torch.zeros(seq_idx.size(0), seq_idx.size(1), len(_TOKENS), dtype=torch.float32)
        x.scatter_(2, seq_idx.unsqueeze(-1).long(), 1.0)
        return x, y
