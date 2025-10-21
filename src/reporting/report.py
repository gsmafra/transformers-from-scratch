"""Utilities to log training metrics to Weights & Biases."""

from typing import Any, Dict

import numpy as np
import torch
import wandb


def _to_numpy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def generate_run_report(run: Any, run_artifacts: Dict[str, Any], prefix: str = "") -> None:
    """Log core training metrics and distributions to W&B.

    If `prefix` is provided, all panel keys are prefixed with it, e.g.,
    "temporal/eval/roc".
    """

    def k(name: str) -> str:
        return f"{prefix}/{name}" if prefix else name

    # Train split
    probabilities = _to_numpy(run_artifacts["probabilities"])
    y_true = _to_numpy(run_artifacts["y"])
    preds = _to_numpy(run_artifacts["predicted_class"])
    # Test split
    probabilities_test = _to_numpy(run_artifacts["probabilities_test"])
    y_true_test = _to_numpy(run_artifacts["y_test"])
    preds_test = _to_numpy(run_artifacts["predicted_class_test"])
    loss_history_train = run_artifacts["loss_history_train"]
    acc_history_train = run_artifacts["accuracy_history_train"]
    loss_history_test = run_artifacts["loss_history_test"]
    acc_history_test = run_artifacts["accuracy_history_test"]
    # Train-only weight/bias histories

    # --- W&B native plots for loss, weight, bias histories ---
    epochs = list(range(len(loss_history_train)))
    epochs_test = list(range(len(loss_history_test)))

    run.log({
        k("metrics/loss"): wandb.plot.line_series(
            xs=epochs,
            ys=[loss_history_train],
            keys=["loss"],
            title="Loss (Train) over epochs",
            xname="epoch",
        )
    })

    weight_history = run_artifacts["weight_history_train"]
    if weight_history:
        # Multi-parameter: one line per parameter (list-of-lists expected)
        n_params = len(weight_history[0])
        ys = [[epoch_vals[j] for epoch_vals in weight_history] for j in range(n_params)]
        keys = [f"w[{j}]" for j in range(n_params)]

        run.log({
            k("metrics/weight"): wandb.plot.line_series(
                xs=epochs,
                ys=ys,
                keys=keys,
                title="Weight over epochs",
                xname="epoch",
            )
        })

    bias_history = run_artifacts["bias_history_train"]
    if bias_history:
        run.log({
            k("metrics/bias"): wandb.plot.line_series(
                xs=epochs,
                ys=[bias_history],
                keys=["bias"],
                title="Bias over epochs",
                xname="epoch",
            )
        })

    # Accuracy and loss line plots
    run.log({
        k("metrics/accuracy_train_history"): wandb.plot.line_series(
            xs=epochs,
            ys=[acc_history_train],
            keys=["accuracy_train"],
            title="Accuracy (Train) over epochs",
            xname="epoch",
        )
    })
    run.log({
        k("metrics/loss_test_history"): wandb.plot.line_series(
            xs=epochs_test,
            ys=[loss_history_test],
            keys=["loss_test"],
            title="Loss (Test) over epochs",
            xname="epoch",
        )
    })
    run.log({
        k("metrics/accuracy_test_history"): wandb.plot.line_series(
            xs=epochs_test,
            ys=[acc_history_test],
            keys=["accuracy_test"],
            title="Accuracy (Test) over epochs",
            xname="epoch",
        )
    })

    # --- Evaluation curves (end-of-run) ---
    if y_true is not None and probabilities is not None:
        # Ensure 1D arrays and build 2D proba matrix for binary classification: [P(class0), P(class1)]
        y_arr = y_true.reshape(-1).astype(int)
        prob_arr = probabilities.reshape(-1).astype(float)
        proba_2d = np.stack([1.0 - prob_arr, prob_arr], axis=1)
        labels = ["0", "1"]
        run.log({k("train/roc"): wandb.plot.roc_curve(y_true=y_arr, y_probas=proba_2d, labels=labels, title="ROC — Train")})
        run.log({k("train/pr"): wandb.plot.pr_curve(y_true=y_arr, y_probas=proba_2d, labels=labels, title="PR — Train")})

    if y_true_test is not None and probabilities_test is not None:
        y_arr = y_true_test.reshape(-1).astype(int)
        prob_arr = probabilities_test.reshape(-1).astype(float)
        proba_2d = np.stack([1.0 - prob_arr, prob_arr], axis=1)
        labels = ["0", "1"]
        run.log({k("test/roc"): wandb.plot.roc_curve(y_true=y_arr, y_probas=proba_2d, labels=labels, title="ROC — Test")})
        run.log({k("test/pr"): wandb.plot.pr_curve(y_true=y_arr, y_probas=proba_2d, labels=labels, title="PR — Test")})

    if y_true is not None and preds is not None:
        y_arr = y_true.reshape(-1).astype(int)
        preds_arr = preds.reshape(-1).astype(int)
        run.log({k("train/confusion"): wandb.plot.confusion_matrix(y_true=y_arr, preds=preds_arr, class_names=["0", "1"], title="Confusion Matrix — Train")})

    if y_true_test is not None and preds_test is not None:
        y_arr = y_true_test.reshape(-1).astype(int)
        preds_arr = preds_test.reshape(-1).astype(int)
        run.log({k("test/confusion"): wandb.plot.confusion_matrix(y_true=y_arr, preds=preds_arr, class_names=["0", "1"], title="Confusion Matrix — Test")})

    # Scalar summaries
    acc_tr = run_artifacts["final_accuracy"]
    loss_tr = run_artifacts["final_loss_train"]
    acc_te = run_artifacts["final_accuracy_test"]
    loss_te = run_artifacts["final_loss_test"]
    scalars = {}
    if isinstance(acc_tr, (int, float)):
        scalars[k("metrics/accuracy_train")] = float(acc_tr)
    if isinstance(loss_tr, (int, float)):
        scalars[k("metrics/loss_train")] = float(loss_tr)
    if isinstance(acc_te, (int, float)):
        scalars[k("metrics/accuracy_test")] = float(acc_te)
    if isinstance(loss_te, (int, float)):
        scalars[k("metrics/loss_test")] = float(loss_te)
    if scalars:
        run.log(scalars)
