import csv
import os
from typing import Any, Dict


CSV_COLUMNS = ["model", "task", "accuracy", "loss"]


def update_benchmark_csv(task: str, results: Dict[str, Dict[str, Any]], *, csv_path: str = "benchmarks/benchmarking.csv") -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    index: Dict[tuple, int] = {}

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(row)
                key = (row.get("model"), row.get("task"))
                index[key] = i

    for model_name, artefacts in results.items():
        acc = artefacts.get("final_accuracy")
        loss = artefacts.get("final_loss")
        key = (model_name, task)
        new_row = {
            "model": model_name,
            "task": task,
            "accuracy": f"{acc:.6f}" if isinstance(acc, (int, float)) else "",
            "loss": f"{loss:.6f}" if isinstance(loss, (int, float)) else "",
        }
        if key in index:
            rows[index[key]] = new_row
        else:
            rows.append(new_row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
