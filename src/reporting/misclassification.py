from html import escape
from typing import List, Optional

import torch


def _format_number(x: float) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _format_numeric_sequence(sample: torch.Tensor) -> str:
    """Format a numeric (non one-hot) sequence by timestep sign/sum.

    Renders the per-timestep summed feature value with sign, compacted to 2 decimals.
    Example: "+0.12 -0.48 0.00 +1.03".
    """
    s = sample.sum(dim=-1).detach().cpu().tolist()  # (T,)
    def fmt(v: float) -> str:
        # Keep a visible sign and small number of decimals
        return f"{v:+.2f}"
    return " ".join(fmt(v) for v in s)


def _sequence_to_str(sample: torch.Tensor, token_names: Optional[List[str]]) -> str:
    """Convert a sequence (T, F) into a readable string for reports.

    - If `token_names` are provided and the sample looks one-hot, render tokens.
    - Otherwise, render numeric per-timestep sums with signs.
    """
    # Heuristic one-hot check: rows sum to 1 and max ~1
    row_sums = sample.sum(dim=-1)
    max_vals, _ = sample.max(dim=-1)
    is_one_hot = bool(
        torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
        and (max_vals > 0.9).all()
    )

    if token_names and is_one_hot:
        idx = sample.argmax(dim=-1).tolist()
        toks = [str(token_names[i]) for i in idx]

        # Smart grouping for arithmetic vocabularies
        arithmetic_syms = set(str(d) for d in range(10)) | {"+", "="}
        if all(t in arithmetic_syms for t in toks):
            parts: List[str] = []
            cur_digits: List[str] = []
            for t in toks:
                if t.isdigit() and len(t) == 1:
                    cur_digits.append(t)
                else:
                    if cur_digits:
                        parts.append("".join(cur_digits))
                        cur_digits = []
                    parts.append(t)
            if cur_digits:
                parts.append("".join(cur_digits))
            return " ".join(parts)
        return " ".join(toks)

    # Numeric fallback
    return _format_numeric_sequence(sample)


def render_misclassified_examples(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    probabilities: torch.Tensor,
    token_names: Optional[List[str]],
    max_wrong: int,
) -> str:
    """Return an HTML section showing a sample of misclassified examples.

    Returns an empty string if inputs are missing or no misclassifications exist.
    """
    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu().reshape(-1).long()
    p_cpu = probabilities.detach().cpu().reshape(-1)
    preds = (p_cpu > 0.5).long()
    wrong_mask = preds.ne(y_cpu)
    wrong_indices = torch.nonzero(wrong_mask, as_tuple=False).reshape(-1).tolist()
    if not wrong_indices:
        return ""

    seen = set()
    rows = []
    for i in wrong_indices:
        seq_idx = tuple(x_cpu[i].argmax(dim=-1).tolist())
        if seq_idx in seen:
            continue
        seen.add(seq_idx)
        seq_str = _sequence_to_str(x_cpu[i], token_names)
        yv = int(y_cpu[i].item())
        pv = float(p_cpu[i].item())
        pr = int(preds[i].item())
        rows.append((seq_idx, i, seq_str, yv, pv, pr))
    rows.sort(key=lambda r: r[0])

    parts: List[str] = []
    parts.append("<div class='section'>")
    parts.append("<h2>Misclassified Examples</h2>")
    parts.append("<p class='subtle'>Sample of inputs the model got wrong at the end of training.</p>")
    parts.append("<table class='tensor'>")
    parts.append("<tr><th>#</th><th>Input</th><th>y_true</th><th>p(class=1)</th><th>pred</th></tr>")

    for _, i, seq_str, yv, pv, pr in rows[:max_wrong]:
        parts.append(
            f"<tr><td>{i}</td><td>{escape(seq_str)}</td><td>{yv}</td><td>{_format_number(pv)}</td><td>{pr}</td></tr>"
        )

    parts.append("</table>")
    parts.append("</div>")
    return "\n".join(parts)
