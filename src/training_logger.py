from typing import Dict, Optional

from torch import Tensor
from wandb.sdk.wandb_run import Run as WandbRun

from .epoch_aggregator import EpochAggregator


class TrainingLogger:
    """Owns per-epoch aggregation and logging for a single model.

    - Aggregates per-batch stats via `EpochAggregator`.
    - Logs epoch summaries to Weights & Biases with model-prefixed keys.
    - Maintains histories used for artifacts reporting.
    """

    def __init__(self, run: WandbRun, model_name: str) -> None:
        self.run = run
        self.model_name = model_name
        self.aggregator = EpochAggregator()
        self.loss_history = []
        self.weight_history = []
        self.bias_history = []

    def start_epoch(self) -> None:
        self.aggregator.reset()

    def update_batch(
        self,
        *,
        logits: Tensor,
        yb: Tensor,
        loss: Tensor,
        grad_norm: float,
        final_linear,
    ) -> None:
        self.aggregator.update(logits=logits, yb=yb, loss=loss, grad_norm=grad_norm, final_linear=final_linear)

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        merged = {f"{self.model_name}/step": epoch}
        for key, value in (metrics or {}).items():
            merged[f"{self.model_name}/metrics/{key}"] = value
        self.run.log(merged)

    def end_epoch(self, epoch: int, extra_metrics: Optional[Dict[str, float]] = None) -> None:
        metrics = self.aggregator.finalize()
        self.loss_history.append(metrics["loss"])

        w_avg = self.aggregator.average_weights()
        b_avg = self.aggregator.average_bias()
        self.weight_history.append(w_avg.view(-1).cpu().tolist())
        self.bias_history.append(float(b_avg))

        if extra_metrics:
            metrics.update(extra_metrics)

        self._log_epoch(epoch, metrics)

    def histories(self) -> Dict[str, list]:
        return {
            "loss_history": self.loss_history,
            "weight_history": self.weight_history,
            "bias_history": self.bias_history,
        }

