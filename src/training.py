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


def train_model(model: ModelAccess, x: Tensor, y: Tensor, run: WandbRun) -> Dict[str, Any]:
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

        # Model-defined extra scalar metrics
        extra = model.extra_metrics(x)
        logger.end_epoch(epoch, extra_metrics=extra)

    with no_grad():
        final_logits = model.forward(x).squeeze(-1)
        final_probabilities = sigmoid(final_logits)
        predicted_class = (final_logits > 0).long()
        accuracy = (predicted_class == y.long().squeeze(-1)).float().mean().item()

    w = final_linear.weight.detach()
    b = final_linear.bias.detach()

    # Export readable artifacts for the trained model
    out_dir = os.path.join("artifacts", "models")
    html_path = export_model_readable_html(
        model,
        out_dir,
        x,
        y,
        final_probabilities,
        token_names=list(ARITH_TOKENS),
        max_wrong=20,
    )

    histories = logger.histories()
    return {
        "x": x.detach(),
        "y": y.detach(),
        "probabilities": final_probabilities.detach(),
        "predicted_class": predicted_class.detach(),
        "loss_history": histories["loss_history"],
        "weight_history": histories["weight_history"],
        "bias_history": histories["bias_history"],
        "final_weight": w,
        "final_bias": b,
        "final_accuracy": accuracy,
        "final_loss": histories["loss_history"][-1] if histories["loss_history"] else None,
        "model_html_path": html_path,
    }


def run_training(
    n_samples: int,
    seed: int,
    run: WandbRun,
    *,
    task: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """High-level convenience function to prepare data, build, and train model."""
    x, y = prepare_data(
        n_samples=n_samples,
        seed=seed,
        task=task,
    )

    # Build the suite of models to train this run
    sequence_length = int(x.size(1))
    models = build_models(sequence_length=sequence_length, n_features=int(x.size(-1)))

    results: Dict[str, Dict[str, Any]] = {}
    for name, mdl in models.items():
        artifacts = train_model(model=mdl, x=x, y=y, run=run)
        # Use wrapper name to key results to avoid mismatch
        results[mdl.name if hasattr(mdl, "name") else name] = artifacts

    return results
