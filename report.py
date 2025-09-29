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

    probabilities = _to_numpy(run_artifacts.get("probabilities"))
    y_true = _to_numpy(run_artifacts.get("y"))
    preds = _to_numpy(run_artifacts.get("predicted_class"))
    loss_history = run_artifacts.get("loss_history", [])
    weight_history = run_artifacts.get("weight_history")
    bias_history = run_artifacts.get("bias_history")

    # --- W&B native plots for loss, weight, bias histories ---
    epochs = list(range(len(loss_history)))

    if loss_history:
        run.log({
            k("metrics/loss"): wandb.plot.line_series(
                xs=epochs,
                ys=[loss_history],
                keys=["loss"],
                title="Loss over epochs",
                xname="epoch",
            )
        })

    if weight_history:
        # Support both 1D (single value per epoch) and 2D (one value per param per epoch)
        first = weight_history[0]
        if isinstance(first, (list, tuple)):
            # Multi-parameter: one line per parameter
            n_params = len(first)
            ys = [[epoch_vals[j] for epoch_vals in weight_history] for j in range(n_params)]
            keys = [f"w[{j}]" for j in range(n_params)]
        else:
            # Backwards-compat: single series
            ys = [weight_history]
            keys = ["weight"]

        run.log({
            k("metrics/weight"): wandb.plot.line_series(
                xs=epochs,
                ys=ys,
                keys=keys,
                title="Weight over epochs",
                xname="epoch",
            )
        })

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

    # --- Evaluation curves (end-of-run) ---
    if y_true is not None and probabilities is not None:
        # Ensure 1D arrays and build 2D proba matrix for binary classification: [P(class0), P(class1)]
        y_arr = y_true.reshape(-1).astype(int)
        prob_arr = probabilities.reshape(-1).astype(float)
        proba_2d = np.stack([1.0 - prob_arr, prob_arr], axis=1)
        labels = ["0", "1"]
        run.log({k("eval/roc"): wandb.plot.roc_curve(y_true=y_arr, y_probas=proba_2d, labels=labels)})
        run.log({k("eval/pr"): wandb.plot.pr_curve(y_true=y_arr, y_probas=proba_2d, labels=labels)})

    if y_true is not None and preds is not None:
        y_arr = y_true.reshape(-1).astype(int)
        preds_arr = preds.reshape(-1).astype(int)
        run.log({
            k("eval/confusion"): wandb.plot.confusion_matrix(y_true=y_arr, preds=preds_arr, class_names=["0", "1"])
        })

    # Distributions are logged during training with proper per-model steps.
