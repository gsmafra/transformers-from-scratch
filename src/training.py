from typing import Any, Dict, Optional

import torch
from torch import Tensor, no_grad, sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from wandb.sdk.wandb_run import Run as WandbRun

from .models import ModelAccess
from .models.registry import build_models
from .tasks import prepare_data


def _log_epoch(run: WandbRun, model_name: str, epoch: int, metrics: Dict[str, float]) -> None:
    merged = {f"{model_name}/step": epoch}
    for key, value in (metrics or {}).items():
        merged[f"{model_name}/metrics/{key}"] = value
    run.log(merged)


def train_model(
    model: ModelAccess,
    x: Tensor,
    y: Tensor,
    run: WandbRun,
) -> Dict[str, Any]:
    """Train the model and return artifacts useful for reporting/analysis."""

    backbone = model.backbone
    criterion = BCEWithLogitsLoss()
    optim = model.make_optimizer()

    loss_history = []
    # Each entry will be a list[float] for all parameters at that epoch
    weight_history = []
    bias_history = []

    # Keep input dimensionality for models to learn from directly.
    # Models are responsible for handling shapes; pass through as-is.
    x_flat = x

    # Final linear for logging/diagnostics
    final_linear = model.final_linear()

    for epoch in trange(model.epochs, desc=f"train:{model.name}", leave=False):
        # Shuffle and iterate over mini-batches for this epoch
        batch_size = getattr(model, "mini_batch_size", None)
        n = int(x_flat.size(0))
        if not batch_size or batch_size <= 0:
            batch_size = n

        indices = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        grad_norm_sum = 0.0
        num_batches = 0
        weight_sum = None
        bias_sum = 0.0
        weight_norm_sum = 0.0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            xb = x_flat[idx]
            yb = y[idx]

            logits = model.forward(xb)
            loss = criterion(logits.squeeze(-1), yb)

            optim.zero_grad()
            loss.backward()
            grad_norm = float(clip_grad_norm_(backbone.parameters(), max_norm=float("inf")).item())
            # Step optimizer per mini-batch
            optim.optimizer.step()

            with no_grad():
                preds = (logits.detach().squeeze(-1) > 0).long()
                total_correct += int((preds == yb.long()).sum().item())
                total_count += int(yb.numel())
                total_loss += float(loss.item()) * int(yb.numel())
                grad_norm_sum += grad_norm
                num_batches += 1
                # Accumulate weights/bias and weight norm for epoch-mean summaries
                base_w = final_linear.weight.detach()
                weight_sum = base_w.clone() if weight_sum is None else weight_sum + base_w
                base_b = final_linear.bias.detach()
                bias_sum += float(base_b.view(-1)[0])
                weight_norm_sum += float(base_w.norm().item())

        # Advance learning rate schedule once per epoch
        optim.scheduler.step()

        loss_epoch = total_loss / max(total_count, 1)
        acc_epoch = float(total_correct) / max(total_count, 1)
        loss_history.append(loss_epoch)

        with no_grad():
            w_avg = (weight_sum / float(num_batches)).detach()
            weight_history.append(w_avg.view(-1).cpu().tolist())
            b_avg = bias_sum / float(num_batches)
            bias_history.append(float(b_avg))
            # Parameter norm averaged over epoch
            weight_norm = float(weight_norm_sum / float(num_batches))

        # Model-defined extra scalar metrics
        extra = model.extra_metrics(x_flat)

        metrics_payload: Dict[str, float] = {
            "loss": float(loss_epoch),
            "accuracy": acc_epoch,
            "grad_norm": float(grad_norm_sum / float(max(num_batches, 1))),
            "weight_norm": weight_norm,
            **(extra or {}),
        }

        _log_epoch(run, model.name, epoch, metrics_payload)

    with no_grad():
        final_logits = model.forward(x_flat).squeeze(-1)
        final_probabilities = sigmoid(final_logits)
        predicted_class = (final_logits > 0).long()
        accuracy = (predicted_class == y.long().squeeze(-1)).float().mean().item()

    w = final_linear.weight.detach()
    b = final_linear.bias.detach()

    return {
        "x": x.detach(),
        "y": y.detach(),
        "probabilities": final_probabilities.detach(),
        "predicted_class": predicted_class.detach(),
        "loss_history": loss_history,
        "weight_history": weight_history,
        "bias_history": bias_history,
        "final_weight": w,
        "final_bias": b,
        "final_accuracy": accuracy,
        "final_loss": loss_history[-1] if loss_history else None,
    }


def run_training(
    sequence_length: int,
    n_samples: int,
    seed: int,
    run: WandbRun,
    *,
    task: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """High-level convenience function to prepare data, build, and train model."""
    x, y = prepare_data(
        sequence_length=sequence_length,
        n_samples=n_samples,
        seed=seed,
        task=task,
    )

    # Build the suite of models to train this run
    models = build_models(sequence_length=sequence_length, n_features=int(x.size(-1)))

    results: Dict[str, Dict[str, Any]] = {}
    for name, mdl in models.items():
        artifacts = train_model(model=mdl, x=x, y=y, run=run)
        # Use wrapper name to key results to avoid mismatch
        results[mdl.name if hasattr(mdl, "name") else name] = artifacts

    return results
