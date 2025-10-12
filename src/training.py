from typing import Any, Callable, Dict, Optional

from tqdm import trange

from torch import Tensor, no_grad, sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


from .models import (
    AttentionAccess,
    LogRegAccess,
    ModelAccess,
    SelfAttentionAccess,
    SelfAttentionQKVAccess,
    TemporalAccess,
)
from .tasks import prepare_data


def train_model(
    model: ModelAccess,
    x: Tensor,
    y: Tensor,
    on_log: Callable[[str, int, Dict[str, float], Any, Any], None],
    *,
    hist_every: int,
) -> Dict[str, Any]:
    """Train the model and return artifacts useful for reporting/analysis."""

    backbone = model.backbone
    criterion = BCEWithLogitsLoss()
    optimizer = AdamW(
        backbone.parameters(),
        lr=model.lr_start,
        weight_decay=getattr(model, "weight_decay", 0.01),
        betas=getattr(model, "betas", (0.9, 0.999)),
        eps=getattr(model, "eps", 1e-8),
    )

    # Linear schedule from lr_start to lr_end across epochs using LambdaLR (multiplicative factors)
    denom = max(model.epochs - 1, 1)
    ratio = float(model.lr_end) / float(model.lr_start + 1e-12)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: 1.0 + (ratio - 1.0) * (ep / denom))

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
        logits = model.forward(x_flat)  # raw logits
        loss = criterion(logits.squeeze(-1), y)

        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        # Gradient norm for diagnostics (no clipping when max_norm=inf)
        grad_norm = float(clip_grad_norm_(backbone.parameters(), max_norm=float("inf")).item())
        optimizer.step()

        with no_grad():
            # Track full weight vector and bias scalar over time
            base_w = final_linear.weight.detach()
            weight_history.append(base_w.view(-1).cpu().tolist())
            base_b = final_linear.bias.detach()
            bias_history.append(float(base_b.view(-1)[0]))

            # Accuracy this epoch
            logits_epoch = logits.detach().squeeze(-1)
            preds_epoch = (logits_epoch > 0).long()
            acc_epoch = (preds_epoch == y.long()).float().mean().item()

            # Parameter norms
            weight_norm = float(final_linear.weight.detach().norm().item())

        # Model-defined extra scalar metrics
        extra = model.extra_metrics(x_flat)

        metrics_payload: Dict[str, float] = {
            "loss": float(loss.item()),
            "accuracy": acc_epoch,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            **(extra or {}),
        }

        on_log(
            model.name,
            epoch,
            metrics_payload,
            None,
            None,
        )

        # Advance LR schedule for next epoch
        scheduler.step()

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
    on_log: Callable[[str, int, Dict[str, float], Any, Any], None],
    *,
    hist_every: int = 10,
    task: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """High-level convenience function to prepare data, build, and train model."""
    x, y = prepare_data(
        sequence_length=sequence_length,
        n_samples=n_samples,
        seed=seed,
        task=task,
    )

    # Align model input dims with generated data
    n_features_eff = int(x.size(-1))

    # Build the suite of models to train this run
    models = {
        "logreg": LogRegAccess(sequence_length=sequence_length, n_features=n_features_eff),
        "temporal": TemporalAccess(sequence_length=sequence_length, n_features=n_features_eff),
        "self_attention": SelfAttentionAccess(n_features=n_features_eff),
        "self_attention_qkv": SelfAttentionQKVAccess(n_features=n_features_eff),
        "attention": AttentionAccess(n_features=n_features_eff),
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, mdl in models.items():
        artifacts = train_model(
            model=mdl,
            x=x,
            y=y,
            on_log=on_log,
            hist_every=hist_every,
        )
        # Use wrapper name to key results to avoid mismatch
        results[mdl.name if hasattr(mdl, "name") else name] = artifacts

    return results
