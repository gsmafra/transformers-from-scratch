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
        # Train metrics (migrated from previous single-set reporting)
        acc_tr = artefacts.get("final_accuracy")
        loss_tr = artefacts.get("final_loss_train")
        key_tr = (model_name, task)
        row_tr = {
            "model": model_name,
            "task": task,
            "accuracy": f"{acc_tr:.6f}" if isinstance(acc_tr, (int, float)) else "",
            "loss": f"{loss_tr:.6f}" if isinstance(loss_tr, (int, float)) else "",
        }
        if key_tr in index:
            rows[index[key_tr]] = row_tr
        else:
            rows.append(row_tr)

        # Test metrics (additional row with explicit split suffix)
        acc_te = artefacts.get("final_accuracy_test")
        loss_te = artefacts.get("final_loss_test")
        if acc_te is not None or loss_te is not None:
            task_te = f"{task} (test)"
            key_te = (model_name, task_te)
            row_te = {
                "model": model_name,
                "task": task_te,
                "accuracy": f"{acc_te:.6f}" if isinstance(acc_te, (int, float)) else "",
                "loss": f"{loss_te:.6f}" if isinstance(loss_te, (int, float)) else "",
            }
            if key_te in index:
                rows[index[key_te]] = row_te
            else:
                rows.append(row_te)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
