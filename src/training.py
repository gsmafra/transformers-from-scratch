import os
from typing import Any, Dict, Optional

import torch
from torch import Tensor, no_grad, sigmoid
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from wandb.sdk.wandb_run import Run as WandbRun

from .reporting.export import export_model_readable_html
from .models import ModelAccess
from .models.registry import build_models
from .tasks import prepare_data
from .tasks.arithmetic_common import TOKENS as ARITH_TOKENS
from .reporting.training_logger import TrainingLogger


def train_model(
    model: ModelAccess,
    x: Tensor,
    y: Tensor,
    run: WandbRun,
    *,
    x_test: Tensor,
    y_test: Tensor,
    task_name: Optional[str],
) -> Dict[str, Any]:
    """Train the model and return artifacts useful for reporting/analysis."""

    backbone = model.backbone
    criterion = model.make_loss()
    optim = model.make_optimizer()

    logger = TrainingLogger(run, model.name)

    # Final linear for logging/diagnostics
    final_linear = model.final_linear()

    for epoch in trange(model.epochs, desc=f"train:{model.name}", leave=False):
        # Shuffle and iterate over mini-batches for this epoch
        batch_size = getattr(model, "mini_batch_size", None)
        n = int(x.size(0))
        if not batch_size or batch_size <= 0:
            batch_size = n

        indices = torch.randperm(n)
        logger.start_epoch()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            xb = x[idx]
            yb = y[idx]

            logits = model.forward(xb)
            loss = criterion(logits.squeeze(-1), yb)

            optim.zero_grad()
            loss.backward()
            grad_norm = float(clip_grad_norm_(backbone.parameters(), max_norm=float("inf")).item())
            # Step optimizer per mini-batch
            optim.optimizer.step()

            logger.update_batch(logits=logits, yb=yb, loss=loss, grad_norm=grad_norm, final_linear=final_linear)

        # Advance learning rate schedule once per epoch
        optim.scheduler.step()

        # End-of-epoch metrics
        extra = model.extra_metrics(x) or {}
        with torch.no_grad():
            logits_te = model.forward(x_test)
            loss_te = criterion(logits_te.squeeze(-1), y_test)
            preds_te = (logits_te.detach().squeeze(-1) > 0).long()
            acc_te = float((preds_te == y_test.long()).float().mean().item())
        logger.end_epoch(
            epoch,
            extra_metrics=extra,
            test_loss=float(loss_te.item()),
            test_accuracy=acc_te,
        )

    with no_grad():
        # Train metrics
        final_logits = model.forward(x).squeeze(-1)
        final_probabilities = sigmoid(final_logits)
        predicted_class = (final_logits > 0).long()
        accuracy = (predicted_class == y.long().squeeze(-1)).float().mean().item()

        # Test metrics
        final_logits_test = model.forward(x_test).squeeze(-1)
        final_probabilities_test = sigmoid(final_logits_test)
        predicted_class_test = (final_logits_test > 0).long()
        accuracy_test = (predicted_class_test == y_test.long().squeeze(-1)).float().mean().item()

    # Export readable artifacts: keep a flat structure under 'artifacts/'
    out_dir = os.path.join("artifacts")
    # Use token names only for tokenized arithmetic tasks (one-hot over 12-token vocab)
    token_names = list(ARITH_TOKENS) if int(x.size(-1)) == len(ARITH_TOKENS) else None

    html_path = export_model_readable_html(
        model,
        out_dir,
        x,
        y,
        final_probabilities,
        token_names=token_names,
        max_wrong=20,
        x_test=x_test,
        y_test=y_test,
        probabilities_test=final_probabilities_test,
        task_name=task_name,
    )

    histories = logger.histories()
    return {
        "x": x.detach(),
        "y": y.detach(),
        "probabilities": final_probabilities.detach(),
        "predicted_class": predicted_class.detach(),
        "accuracy_history_train": histories["accuracy_history_train"],
        "loss_history_test": histories["loss_history_test"],
        "loss_history_train": histories["loss_history_train"],
        "accuracy_history_test": histories["accuracy_history_test"],
        "x_test": x_test.detach(),
        "y_test": y_test.detach(),
        "probabilities_test": final_probabilities_test.detach(),
        "predicted_class_test": predicted_class_test.detach(),
        "weight_history_train": histories["weight_history_train"],
        "bias_history_train": histories["bias_history_train"],
        "final_accuracy": accuracy,
        "final_accuracy_test": accuracy_test,
        "final_loss_train": histories["loss_history_train"][-1] if histories["loss_history_train"] else None,
        "final_loss_test": float(model.make_loss()(final_logits_test, y_test).item()),
        "model_html_path": html_path,
    }


def run_training(
    n_samples: int,
    seed: int,
    run: WandbRun,
    *,
    task: str,
    model_names: list[str],
) -> Dict[str, Dict[str, Any]]:
    """High-level convenience function to prepare data, build, and train model."""
    x, y = prepare_data(n_samples=n_samples, seed=seed, task=task)
    # Prepare test split with a different seed
    x_test, y_test = prepare_data(n_samples=n_samples, seed=seed + 1, task=task)

    # Build the suite of models to train this run
    sequence_length = int(x.size(1))
    models = build_models(sequence_length=sequence_length, n_features=int(x.size(-1)), only=model_names)

    results: Dict[str, Dict[str, Any]] = {}
    for name, mdl in models.items():
        artifacts = train_model(
            model=mdl,
            x=x,
            y=y,
            run=run,
            x_test=x_test,
            y_test=y_test,
            task_name=task,
        )
        # Use wrapper name to key results to avoid mismatch
        results[mdl.name if hasattr(mdl, "name") else name] = artifacts

    return results
