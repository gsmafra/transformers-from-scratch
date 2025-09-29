from typing import Any, Callable, Dict, Tuple
from abc import ABC, abstractmethod

from torch import logit, no_grad, tanh, Tensor
from torch.nn import BCELoss, Linear, Sequential, Sigmoid, Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD

from data import prepare_data

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


# --- Model accessors -------------------------------------------------------

class ModelAccess(ABC):
    """Lightweight adapter exposing a unified interface to the training loop."""

    name: str
    backbone: Module
    epochs: int
    lr: float

    def __init__(self, name: str, backbone: Module, *, epochs: int, lr: float) -> None:
        self.name = name
        self.backbone = backbone
        self.epochs = epochs
        self.lr = lr

    def forward(self, x_flat: Tensor) -> Tensor:
        return self.backbone(x_flat)

    @abstractmethod
    def final_linear(self) -> Linear:
        """Return the final Linear layer used for weight/bias tracking."""
        raise NotImplementedError


class LogRegAccess(ModelAccess):
    def __init__(self, sequence_length: int, *, epochs: int = 1000, lr: float = 1.0) -> None:
        super().__init__(
            name="logreg",
            backbone=build_logreg(sequence_length),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone[0]  # type: ignore[index]


class TemporalAccess(ModelAccess):
    def __init__(self, sequence_length: int, *, epochs: int = 1000, lr: float = 1.0) -> None:
        super().__init__(
            name="temporal",
            backbone=build_model(sequence_length),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[0]  # type: ignore[attr-defined,index]


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
    criterion = BCELoss()
    optimizer = SGD(backbone.parameters(), lr=model.lr)

    loss_history = []
    # Each entry will be a list[float] for all parameters at that epoch
    weight_history = []
    bias_history = []

    # Flatten sequence dimension for downstream layers
    x_flat = x.view(x.size(0), -1)

    # Final linear for logging/diagnostics
    final_linear = model.final_linear()

    for epoch in range(model.epochs):
        pred = model.forward(x_flat)  # probabilities
        loss = criterion(pred.squeeze(-1), y)

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
            probs_epoch = pred.detach().squeeze(-1)
            preds_epoch = (probs_epoch > 0.5).long()
            acc_epoch = (preds_epoch == y.long()).float().mean().item()

            # Parameter norms
            weight_norm = float(final_linear.weight.detach().norm().item())
            bias_abs = float(final_linear.bias.detach().abs().mean().item())

        # Prepare optional distributions for logging at cadence
        probs_np = None
        logits_np = None
        if (epoch % hist_every == 0 or epoch == model.epochs - 1):
            with no_grad():
                probs_batch = model.forward(x_flat).squeeze(-1)
                probs_np = probs_batch.detach().cpu().numpy()
                # Derive logits from probabilities to avoid model-specific access
                logits_np = logit(probs_batch.clamp(1e-6, 1 - 1e-6)).detach().cpu().numpy()

        on_log(
            model.name,
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
        final_probabilities = model.forward(x_flat).squeeze(-1)
        predicted_class = (final_probabilities > 0.5).long()
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
) -> Dict[str, Dict[str, Any]]:
    """High-level convenience function to prepare data, build, and train model."""
    x, y = prepare_data(sequence_length=sequence_length, n_samples=n_samples, seed=seed)

    # Build the suite of models to train this run
    models = {
        "logreg": LogRegAccess(sequence_length=sequence_length),
        "temporal": TemporalAccess(sequence_length=sequence_length),
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
