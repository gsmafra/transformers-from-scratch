from html import escape
from typing import List, Optional

import torch


def _format_number(x: float) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def compute_vocab_self_attention(model, token_names: Optional[List[str]]) -> Optional[torch.Tensor]:
    """Return a (V, V) attention-like matrix over the vocabulary for supported models.

    Uses simple projections available on the model to estimate attention weights.
    Returns None if not applicable.
    """
    if not token_names:
        return None
    bb = model.backbone
    with torch.no_grad():
        if hasattr(bb, "proj") and hasattr(bb, "d_model"):
            W = bb.proj.weight.detach().cpu()  # (d, V)
            b = bb.proj.bias.detach().cpu()  # (d)
            E = torch.tanh(W.T + b)  # (V, d)
            scale = float(bb.d_model) ** 0.5
            logits = (E @ E.T) / scale  # (V, V)
            return torch.softmax(logits, dim=1)
        if hasattr(bb, "q_proj") and hasattr(bb, "k_proj") and hasattr(bb, "d_model"):
            Wq = bb.q_proj.weight.detach().cpu()  # (d, V)
            bq = bb.q_proj.bias.detach().cpu()
            Wk = bb.k_proj.weight.detach().cpu()
            bk = bb.k_proj.bias.detach().cpu()
            Q = torch.tanh(Wq.T + bq)  # (V, d)
            K = torch.tanh(Wk.T + bk)  # (V, d)
            scale = float(bb.d_model) ** 0.5
            logits = (Q @ K.T) / scale  # (V, V)
            return torch.softmax(logits, dim=1)
    return None


def render_vocab_attention_section(model, token_names: Optional[List[str]]) -> str:
    """Return an HTML section rendering the vocab self-attention matrix, or empty string."""
    attn = compute_vocab_self_attention(model, token_names)
    if attn is None:
        return ""

    V = attn.size(0)
    headers = token_names if token_names and len(token_names) == V else [f"c{i}" for i in range(V)]
    rows = token_names if token_names and len(token_names) == V else [f"r{i}" for i in range(V)]

    parts: List[str] = []
    parts.append("<div class='section'>")
    parts.append("<h2>Vocabulary Self-Attention</h2>")

    parts.append("<table class='tensor'>")
    parts.append("<tr><th></th>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr>")

    data = attn.detach().cpu().tolist()
    for r in range(V):
        parts.append(
            f"<tr><th>{escape(str(rows[r]))}</th>" + "".join(f"<td>{_format_number(val)}</td>" for val in data[r]) + "</tr>"
        )
    parts.append("</table>")
    parts.append("</div>")
    return "\n".join(parts)
