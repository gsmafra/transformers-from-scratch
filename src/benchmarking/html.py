import csv
import os
from typing import Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape


def generate_benchmark_html(
    *,
    csv_path: str = "benchmarks/benchmarking.csv",
    html_path: str = "benchmarks/index.html",
    title: str = "Model × Task Benchmarks",
    overlap_csv_path: str = "benchmarks/overlap.csv",
) -> None:
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    rows: List[dict] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Split train vs test by task naming convention: "<task> (test)"
    def is_test_task(task_name: str) -> Tuple[bool, str]:
        task_name = task_name or ""
        suffix = " (test)"
        if task_name.endswith(suffix):
            return True, task_name[: -len(suffix)]
        return False, task_name

    models = sorted({r.get("model", "") for r in rows if r.get("model")})
    tasks_train_set = set()
    tasks_test_set = set()
    values_train: Dict[str, Dict[str, Dict[str, str]]] = {}
    values_test: Dict[str, Dict[str, Dict[str, str]]] = {}

    for r in rows:
        m = r.get("model", "") or ""
        t_full = r.get("task", "") or ""
        if not m or not t_full:
            continue
        is_test, t_base = is_test_task(t_full)
        target = values_test if is_test else values_train
        (tasks_test_set if is_test else tasks_train_set).add(t_base)
        target.setdefault(m, {})[t_base] = {
            "accuracy": (r.get("accuracy", "") or ""),
            "loss": (r.get("loss", "") or ""),
        }

    tasks_train = sorted(tasks_train_set)
    tasks_test = sorted(tasks_test_set)

    def _compute_averages(values: Dict[str, Dict[str, Dict[str, str]]], tasks: List[str]) -> Dict[str, Dict[str, str]]:
        avgs: Dict[str, Dict[str, str]] = {}
        for m, per_task in values.items():
            accs: List[float] = []
            losses: List[float] = []
            for t in tasks:
                cell = per_task.get(t)
                if not cell:
                    continue
                a = cell.get("accuracy") or ""
                l = cell.get("loss") or ""
                try:
                    if a != "":
                        accs.append(float(a))
                except ValueError:
                    pass
                try:
                    if l != "":
                        losses.append(float(l))
                except ValueError:
                    pass
            acc_avg = sum(accs) / len(accs) if accs else None
            loss_avg = sum(losses) / len(losses) if losses else None
            avgs[m] = {
                "accuracy": (f"{acc_avg:.6f}" if acc_avg is not None else ""),
                "loss": (f"{loss_avg:.6f}" if loss_avg is not None else ""),
            }
        return avgs

    avg_train = _compute_averages(values_train, tasks_train)
    avg_test = _compute_averages(values_test, tasks_test)

    # Load overlap stats if available
    overlap_rows: List[dict] = []
    if os.path.exists(overlap_csv_path):
        with open(overlap_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            overlap_rows = list(reader)

    # Render using Jinja2 template; templates live at repo root under 'templates'
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    templates_dir = os.path.join(repo_root, "templates")
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=False,
    )
    template = env.get_template("benchmarks.html")
    output = template.render(
        title=title,
        models=models,
        tasks_train=tasks_train,
        tasks_test=tasks_test,
        values_train=values_train,
        values_test=values_test,
        avg_train=avg_train,
        avg_test=avg_test,
        overlap_rows=overlap_rows,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(output)
