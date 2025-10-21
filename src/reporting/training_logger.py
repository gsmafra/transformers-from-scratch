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
        self.loss_history_train = []
        self.accuracy_history_train = []
        self.weight_history_train = []
        self.bias_history_train = []
        self.loss_history_test = []
        self.accuracy_history_test = []

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

    def end_epoch(
        self,
        epoch: int,
        *,
        extra_metrics: Optional[Dict[str, float]] = None,
        test_loss: Optional[float] = None,
        test_accuracy: Optional[float] = None,
    ) -> None:
        metrics = self.aggregator.finalize()
        # Track train histories
        self.loss_history_train.append(metrics["loss"])
        self.accuracy_history_train.append(metrics["accuracy"])

        w_avg = self.aggregator.average_weights()
        b_avg = self.aggregator.average_bias()
        self.weight_history_train.append(w_avg.view(-1).cpu().tolist())
        self.bias_history_train.append(float(b_avg))

        # Remap core metrics to explicit train keys
        out_metrics: Dict[str, float] = {
            "loss_train": metrics.get("loss", 0.0),
            "accuracy_train": metrics.get("accuracy", 0.0),
            "grad_norm": metrics.get("grad_norm", 0.0),
            "weight_norm": metrics.get("weight_norm", 0.0),
        }

        # Merge in model-specific metrics only
        if extra_metrics:
            out_metrics.update(extra_metrics)

        # Add and record test metrics (global across models)
        if test_loss is not None:
            self.loss_history_test.append(float(test_loss))
            out_metrics["loss_test"] = float(test_loss)
        if test_accuracy is not None:
            self.accuracy_history_test.append(float(test_accuracy))
            out_metrics["accuracy_test"] = float(test_accuracy)

        self._log_epoch(epoch, out_metrics)

    def histories(self) -> Dict[str, list]:
        return {
            "loss_history_train": self.loss_history_train,
            "accuracy_history_train": self.accuracy_history_train,
            "weight_history_train": self.weight_history_train,
            "bias_history_train": self.bias_history_train,
            "loss_history_test": self.loss_history_test,
            "accuracy_history_test": self.accuracy_history_test,
        }
