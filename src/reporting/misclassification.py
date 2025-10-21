from html import escape
from typing import List, Optional

import torch


def _format_number(x: float) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _one_hot_to_token_str(sample: torch.Tensor, token_names: Optional[List[str]]) -> str:
    """Convert a one-hot sequence (T, V) into a readable token string.

    - Uses `token_names` when provided.
    - If tokens look like arithmetic digits/operators, groups consecutive digits
      into multi-digit numbers and spaces around '+', '='.
    - Otherwise falls back to space-separated tokens.
    """
    idx = sample.argmax(dim=-1).tolist()
    if token_names and len(token_names) >= sample.size(-1):
        toks = [str(token_names[i]) for i in idx]
    else:
        toks = [str(i) for i in idx]

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

    # Fallback: space-separated tokens
    return " ".join(toks)


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
        seq_str = _one_hot_to_token_str(x_cpu[i], token_names)
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
