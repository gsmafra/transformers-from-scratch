from typing import Any, Callable, Dict, Tuple

import torch
from torch import logit, manual_seed, no_grad, randn, tanh, Tensor
from torch.nn import BCELoss, Linear, Sequential, Sigmoid, Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD


def prepare_data(sequence_length: int, n_samples: int, seed: int = 0) -> Tuple[Tensor, Tensor]:
    """Generate random normal sequences X and labels y = 1[any x > 0].

    Returns:
    - x: (n_samples, sequence_length, 1)
    - y: (n_samples,)
    """
    manual_seed(seed)

    # Skew the per-timestep distribution negative so that P(any x > 0) ≈ 0.5.
    # For per-timestep positive rate p and sequence length T: 1 - (1 - p)^T ≈ 0.5
    p_target = 1.0 - 0.5 ** (1.0 / float(sequence_length))
    mu = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p_target)).item()

    x = randn(n_samples, sequence_length, 1) + mu
    y = (x.squeeze(-1).max(dim=1).values > 0).float()
    # Inject label noise: flip label with 10% probability per sample
    flip = (torch.rand(n_samples) < 0.10).float()
    y = torch.where(flip > 0, 1.0 - y, y)
    return x, y


class SimpleTemporalPoolingClassifier(Module):
    def __init__(self, sequence_length: int, d_model: int = 16) -> None:
        super().__init__()
        self.proj = Linear(1, d_model)
        # Classifier operates over timesteps (length T)
        self.classifier = Sequential(Linear(sequence_length, 1), Sigmoid())

    def forward(self, x_flat: Tensor) -> Tensor:
        # x_flat: (N, T)
        h = tanh(self.proj(x_flat.unsqueeze(-1)))  # (N, T, d)
        # Compress feature dimension, keep timesteps
        pooled = h.mean(dim=-1)  # (N, T)
        return self.classifier(pooled)  # (N, 1)


def build_model(sequence_length: int) -> Module:
    # Simple projection per timestep, compress features, then classify over T
    return SimpleTemporalPoolingClassifier(sequence_length=sequence_length, d_model=16)


def build_logreg(sequence_length: int) -> Module:
    """Baseline logistic regression over flattened sequence."""
    return Sequential(Linear(sequence_length, 1), Sigmoid())


def train_model(
    model: Module,
    x: Tensor,
    y: Tensor,
    epochs: int,
    learning_rate: float,
    model_name: str,
    on_log: Callable[[str, int, Dict[str, float], Any, Any], None],
    hist_every: int = 10,
) -> Dict[str, Any]:
    """Train the model and return artifacts useful for reporting/analysis."""

    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    loss_history = []
    # Each entry will be a list[float] for all parameters at that epoch
    weight_history = []
    bias_history = []

    # Flatten sequence dimension for downstream layers
    x_flat = x.view(x.size(0), -1)

    # Identify the final linear layer for logging (supports both models)
    if hasattr(model, "classifier") and isinstance(model.classifier, Sequential):
        linear_layer = model.classifier[0]
    elif isinstance(model, Sequential):
        linear_layer = model[0]
    else:
        linear_layer = None

    for epoch in range(epochs):
        pred = model(x_flat)  # probabilities
        loss = criterion(pred.squeeze(-1), y)

        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        # Gradient norm for diagnostics (no clipping when max_norm=inf)
        grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=float("inf")).item())
        optimizer.step()

        with no_grad():
            # Track full weight vector and bias scalar over time
            if linear_layer is not None:
                base_w = linear_layer.weight.detach()
                weight_history.append(base_w.view(-1).cpu().tolist())
                base_b = linear_layer.bias.detach()
                bias_history.append(float(base_b.view(-1)[0]))
            else:
                weight_history.append([])
                bias_history.append(0.0)

            # Accuracy this epoch
            probs_epoch = pred.detach().squeeze(-1)
            preds_epoch = (probs_epoch > 0.5).long()
            acc_epoch = (preds_epoch == y.long()).float().mean().item()

            # Parameter norms
            if linear_layer is not None:
                weight_norm = float(linear_layer.weight.detach().norm().item())
                bias_abs = float(linear_layer.bias.detach().abs().mean().item())
            else:
                weight_norm = 0.0
                bias_abs = 0.0

        # Prepare optional distributions for logging at cadence
        probs_np = None
        logits_np = None
        if (epoch % hist_every == 0 or epoch == epochs - 1):
            with no_grad():
                probs_batch = model(x_flat).squeeze(-1)
                probs_np = probs_batch.detach().cpu().numpy()
                # Derive logits from probabilities to avoid model-specific access
                logits_np = logit(probs_batch.clamp(1e-6, 1 - 1e-6)).detach().cpu().numpy()

        on_log(
            model_name,
            epoch,
            {
                "loss": float(loss.item()),
                "accuracy": acc_epoch,
                "grad_norm": grad_norm,
                "weight_norm": weight_norm,
                "bias_abs": bias_abs,
            },
            probs_np,
            logits_np,
        )

    with no_grad():
        final_probabilities = model(x_flat).squeeze(-1)
        predicted_class = (final_probabilities > 0.5).long()
        accuracy = (predicted_class == y.long().squeeze(-1)).float().mean().item()

    if linear_layer is not None:
        w = linear_layer.weight.detach()
        b = linear_layer.bias.detach()
    else:
        w = x.new_empty(0)
        b = x.new_empty(0)

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
    epochs: int,
    learning_rate: float,
    sequence_length: int,
    n_samples: int,
    seed: int,
    on_log: Callable[[str, int, Dict[str, float], Any, Any], None],
    hist_every: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """High-level convenience function to prepare data, build, and train model."""
    x, y = prepare_data(sequence_length=sequence_length, n_samples=n_samples, seed=seed)
    # Train baseline logistic regression
    logreg = build_logreg(sequence_length=sequence_length)
    artifacts_logreg = train_model(
        model=logreg,
        x=x,
        y=y,
        epochs=epochs,
        learning_rate=learning_rate,
        model_name="logreg",
        on_log=on_log,
        hist_every=hist_every,
    )
    
    # Train temporal pooling model
    temporal = build_model(sequence_length=sequence_length)
    artifacts_temporal = train_model(
        model=temporal,
        x=x,
        y=y,
        epochs=epochs,
        learning_rate=learning_rate,
        model_name="temporal",
        on_log=on_log,
        hist_every=hist_every,
    )

    return {"logreg": artifacts_logreg, "temporal": artifacts_temporal}
