import os
from html import escape
from typing import List, Optional

import torch

from ..models.base import ModelAccess
from .attention_viz import render_vocab_attention_section
from .misclassification import render_misclassified_examples


def _load_css() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    css_path = os.path.join(repo_root, "templates", "model_report.css")
    with open(css_path, "r", encoding="utf-8") as cf:
        css = cf.read()
    return f"<style>\n{css}\n</style>"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def build_model_readable_html(
    model: ModelAccess,
    x: torch.Tensor,
    y: torch.Tensor,
    probabilities: torch.Tensor,
    token_names: Optional[List[str]] = None,
    max_wrong: int = 20,
    *,
    x_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    probabilities_test: Optional[torch.Tensor] = None,
) -> str:
    """Build the HTML string for the model report (no I/O)."""
    total_params = sum(p.numel() for p in model.backbone.parameters())
    param_meta = [(n, tuple(p.shape), p.numel()) for n, p in model.backbone.named_parameters()]

    # Load CSS from templates for readability and syntax highlighting; embed inline for portability
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

    # Misclassified examples (train)
    misc_train = render_misclassified_examples(
        x=x, y=y, probabilities=probabilities, token_names=token_names, max_wrong=max_wrong
    )
    if misc_train:
        html.append(misc_train.replace("<h2>Misclassified Examples</h2>", "<h2>Misclassified Examples — Train</h2>"))

    # Misclassified examples (test)
    if x_test is not None and y_test is not None and probabilities_test is not None:
        misc_test = render_misclassified_examples(
            x=x_test, y=y_test, probabilities=probabilities_test, token_names=token_names, max_wrong=max_wrong
        )
        if misc_test:
            html.append(misc_test.replace("<h2>Misclassified Examples</h2>", "<h2>Misclassified Examples — Test</h2>"))

    # Vocabulary self-attention (for attention models)
    attn_html = render_vocab_attention_section(model, token_names)
    if attn_html:
        html.append(attn_html)

    html.append("</body></html>")
    return "\n".join(html)


def export_model_readable_html(
    model: ModelAccess,
    dir_path: str,
    x: torch.Tensor,
    y: torch.Tensor,
    probabilities: torch.Tensor,
    token_names: Optional[List[str]] = None,
    max_wrong: int = 20,
    *,
    x_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    probabilities_test: Optional[torch.Tensor] = None,
) -> str:
    """Export a single self-contained HTML report with architecture and weights.

    Returns the file path written.
    """
    _ensure_dir(dir_path)
    path = os.path.join(dir_path, f"{model.name}_report.html")
    html_content = build_model_readable_html(
        model,
        x,
        y,
        probabilities,
        token_names,
        max_wrong,
        x_test=x_test,
        y_test=y_test,
        probabilities_test=probabilities_test,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return path
