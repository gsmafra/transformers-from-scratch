import csv
import os
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape


def generate_benchmark_html(
    *,
    csv_path: str = "benchmarks/benchmarking.csv",
    html_path: str = "benchmarks/index.html",
    title: str = "Model × Task Benchmarks",
) -> None:
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    rows: List[dict] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Compute pivot structures
    models = sorted({r.get("model", "") for r in rows if r.get("model")})
    tasks = sorted({r.get("task", "") for r in rows if r.get("task")})

    values: Dict[str, Dict[str, Dict[str, str]]] = {}
    for r in rows:
        m = r.get("model", "") or ""
        t = r.get("task", "") or ""
        if not m or not t:
            continue
        values.setdefault(m, {})[t] = {
            "accuracy": (r.get("accuracy", "") or ""),
            "loss": (r.get("loss", "") or ""),
        }

    # Render using Jinja2 template
    base_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(base_dir, "templates")
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=False,
    )
    template = env.get_template("benchmarks.html.j2")
    output = template.render(title=title, models=models, tasks=tasks, values=values)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(output)
