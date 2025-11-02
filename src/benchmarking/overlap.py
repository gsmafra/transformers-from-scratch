import csv
import os
from typing import Any, Dict

from torch import Tensor


CSV_COLUMNS = ["task", "overlap_count", "test_total", "overlap_rate"]


def _row_signature(x_row: Tensor) -> tuple:
    """Return a hashable signature for a single sample (T,F) tensor.

    Uses exact float tuples; appropriate for synthetic datasets here.
    """
    return tuple(float(v) for v in x_row.reshape(-1).tolist())


def count_overlap(x_train: Tensor, x_test: Tensor) -> Dict[str, Any]:
    """Count how many test samples also appear in train (exact match over all features)."""
    train_set = { _row_signature(x_train[i]) for i in range(int(x_train.size(0))) }
    hits = 0
    n_test = int(x_test.size(0))
    for i in range(n_test):
        sig = _row_signature(x_test[i])
        if sig in train_set:
            hits += 1
    rate = float(hits) / float(n_test) if n_test > 0 else 0.0
    return {"overlap_count": hits, "test_total": n_test, "overlap_rate": rate}


def update_overlap_csv(*, task: str, x: Tensor, x_test: Tensor, csv_path: str = "benchmarks/overlap.csv") -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    index: Dict[str, int] = {}

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(row)
                key = row.get("task")
                if key:
                    index[key] = i

    stats = count_overlap(x, x_test)
    row = {
        "task": task,
        "overlap_count": str(int(stats["overlap_count"])),
        "test_total": str(int(stats["test_total"])),
        "overlap_rate": f"{float(stats["overlap_rate"]):.6f}",
    }
    if task in index:
        rows[index[task]] = row
    else:
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

