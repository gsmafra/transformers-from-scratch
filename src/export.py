import json
import os
from html import escape
from typing import Dict, List, Optional

import torch

from .models.base import ModelAccess


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_model_definition(model: ModelAccess, dir_path: str) -> str:
    """Write a readable text file describing the model architecture.

    Returns the file path written.
    """
    _ensure_dir(dir_path)
    path = os.path.join(dir_path, f"{model.name}_architecture.txt")
    # Use the module's string representation for a concise, readable structure
    with open(path, "w", encoding="utf-8") as f:
        f.write(repr(model.backbone))
        f.write("\n\n")
        # Add parameter counts by name
        total_params = 0
        for name, param in model.backbone.named_parameters():
            numel = param.numel()
            total_params += numel
            f.write(f"{name}: shape={tuple(param.shape)} numel={numel}\n")
        f.write(f"\nTotal parameters: {total_params}\n")
    return path


def export_model_weights(model: ModelAccess, dir_path: str) -> str:
    """Write all model weights to JSON for readability.

    Returns the file path written.
    """
    _ensure_dir(dir_path)
    path = os.path.join(dir_path, f"{model.name}_weights.json")
    state: Dict[str, torch.Tensor] = model.backbone.state_dict()
    serializable = {k: v.detach().cpu().tolist() for k, v in state.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)
    return path


def _format_number(x: float) -> str:
    try:
        # Compact formatting for readability
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _tensor_to_html_table(
    name: str,
    t: torch.Tensor,
    token_names: Optional[List[str]] = None,
    *,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
) -> str:
    t = t.detach().cpu()
    shape = list(t.shape)
    dim = t.dim()
    html_parts: List[str] = []
    html_parts.append(f"<h3>{escape(name)} — shape {escape(str(shape))}</h3>")
    if dim == 0:
        html_parts.append(f"<table class='tensor'><tr><td>{_format_number(t.item())}</td></tr></table>")
    elif dim == 1:
        html_parts.append("<table class='tensor'>")
        html_parts.append("<tr>" + "".join(f"<th>{i}</th>" for i in range(shape[0])) + "</tr>")
        html_parts.append("<tr>" + "".join(f"<td>{_format_number(v)}</td>" for v in t.tolist()) + "</tr>")
        html_parts.append("</table>")
    elif dim == 2:
        rows, cols = shape
        html_parts.append("<table class='tensor'>")
        # header
        headers = None
        if col_labels is not None and len(col_labels) == cols:
            headers = col_labels
        elif token_names and cols == len(token_names) and ("proj" in name):
            headers = token_names
        if headers is None:
            headers = [f"c{i}" for i in range(cols)]
        html_parts.append("<tr><th></th>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr>")
        data = t.tolist()
        for r in range(rows):
            if row_labels is not None and len(row_labels) == rows:
                row_name = str(row_labels[r])
            else:
                row_name = f"r{r}"
            html_parts.append(f"<tr><th>{escape(row_name)}</th>" + "".join(f"<td>{_format_number(val)}</td>" for val in data[r]) + "</tr>")
        html_parts.append("</table>")
    else:
        # For higher dims, show the first dimension slices
        first = shape[0]
        for i in range(first):
            html_parts.append(f"<h4>slice {i} along dim0</h4>")
            html_parts.append(_tensor_to_html_table(f"{name}[{i}]", t[i], token_names=token_names).replace("<h3", "<h5").replace("</h3>", "</h5>"))
    return "\n".join(html_parts)


def _compute_vocab_self_attention(model: ModelAccess, token_names: Optional[List[str]]) -> Optional[torch.Tensor]:
    if not token_names:
        return None
    V = len(token_names)
    name = getattr(model, "name", "")
    bb = model.backbone
    with torch.no_grad():
        if name == "self_attention" and hasattr(bb, "proj") and hasattr(bb, "d_model"):
            # E: (V, d)
            W = bb.proj.weight.detach().cpu()  # (d, V)
            b = bb.proj.bias.detach().cpu()  # (d)
            E = torch.tanh(W.T + b)  # (V, d)
            scale = float(bb.d_model) ** 0.5
            logits = (E @ E.T) / scale  # (V, V)
            attn = torch.softmax(logits, dim=1)
            return attn
        if name == "self_attention_qkv" and hasattr(bb, "q_proj") and hasattr(bb, "k_proj") and hasattr(bb, "d_model"):
            Wq = bb.q_proj.weight.detach().cpu()  # (d, V)
            bq = bb.q_proj.bias.detach().cpu()
            Wk = bb.k_proj.weight.detach().cpu()
            bk = bb.k_proj.bias.detach().cpu()
            Q = torch.tanh(Wq.T + bq)  # (V, d)
            K = torch.tanh(Wk.T + bk)  # (V, d)
            scale = float(bb.d_model) ** 0.5
            logits = (Q @ K.T) / scale  # (V, V)
            attn = torch.softmax(logits, dim=1)
            return attn
    return None


def export_model_readable_html(model: ModelAccess, dir_path: str, token_names: Optional[List[str]] = None) -> str:
    """Export a single self-contained HTML report with architecture and weights.

    Returns the file path written.
    """
    _ensure_dir(dir_path)
    path = os.path.join(dir_path, f"{model.name}_report.html")

    total_params = sum(p.numel() for p in model.backbone.parameters())
    param_meta = [(n, tuple(p.shape), p.numel()) for n, p in model.backbone.named_parameters()]

    # Load CSS from templates for readability and syntax highlighting; embed inline for portability
    def _load_css() -> str:
        repo_root = os.path.dirname(os.path.dirname(__file__))
        css_path = os.path.join(repo_root, "templates", "model_report.css")
        try:
            with open(css_path, "r", encoding="utf-8") as cf:
                css = cf.read()
        except FileNotFoundError:
            css = (
                ":root{--bg:#0b0f14;--panel:#0f1620;--text:#e6edf3;--muted:#9aa4b2;--border:#253244;--header:#17212b;}"
                "body{background:var(--bg);color:var(--text);font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;margin:20px;}"
                "table{border-collapse:collapse;}table th,table td{border:1px solid var(--border);padding:4px 6px;}"
            )
        return f"<style>\n{css}\n</style>"
    styles = _load_css()

    html: List[str] = []
    html.append("<!DOCTYPE html>")
    html.append("<html><head><meta charset='utf-8'>")
    html.append(f"<title>{escape(model.name)} — Model Report</title>")
    html.append(styles)
    html.append("</head><body>")
    html.append(f"<h1>{escape(model.name)}</h1>")
    html.append(f"<div class='subtle'>Total parameters: {total_params}</div>")

    # Architecture
    html.append("<div class='section'>")
    html.append("<h2>Architecture</h2>")
    html.append("<p class='subtle'>Module tree (via repr):</p>")
    html.append(f"<pre>{escape(repr(model.backbone))}</pre>")
    # Parameter meta
    html.append("<table class='param-table'>")
    html.append("<tr><th>Parameter</th><th>Shape</th><th>Numel</th></tr>")
    for name, shape, numel in param_meta:
        html.append(f"<tr><td>{escape(name)}</td><td>{escape(str(shape))}</td><td>{numel}</td></tr>")
    html.append("</table>")
    html.append("</div>")

    # Note: token_names, when provided, are used to annotate column headers
    # of projection weight matrices (e.g., 'proj.weight', 'q_proj.weight', etc.).

    # Weights
    html.append("<div class='section'>")
    html.append("<h2>Weights</h2>")
    for name, tensor in model.backbone.state_dict().items():
        html.append(_tensor_to_html_table(name, tensor, token_names=token_names))
    html.append("</div>")

    # Vocabulary self-attention (for attention models)
    attn = _compute_vocab_self_attention(model, token_names)
    if attn is not None:
        html.append("<div class='section'>")
        html.append("<h2>Vocabulary Self-Attention</h2>")
        html.append(_tensor_to_html_table(
            "attn[vocab→vocab] (softmax over columns)",
            attn,
            row_labels=token_names,
            col_labels=token_names,
        ))
        html.append("</div>")

    html.append("</body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return path
