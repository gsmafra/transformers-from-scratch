from typing import Dict

from torch import Tensor, no_grad


class EpochAggregator:
    """Accumulates per-batch training statistics and computes epoch summaries.

    Tracks sample-weighted loss and accuracy, mean grad norm, and averages of
    the final layer's weights, bias, and weight norm across batches.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_count = 0
        self.grad_norm_sum = 0.0
        self.num_batches = 0
        self._weight_sum = None
        self._bias_sum = 0.0
        self._weight_norm_sum = 0.0

    @no_grad()
    def update(self, *, logits: Tensor, yb: Tensor, loss: Tensor, grad_norm: float, final_linear) -> None:
        preds = (logits.detach().squeeze(-1) > 0).long()
        self.total_correct += int((preds == yb.long()).sum().item())
        n = int(yb.numel())
        self.total_count += n
        self.total_loss += float(loss.item()) * n
        self.grad_norm_sum += float(grad_norm)
        self.num_batches += 1

        base_w = final_linear.weight.detach()
        self._weight_sum = base_w.clone() if self._weight_sum is None else self._weight_sum + base_w
        base_b = final_linear.bias.detach()
        self._bias_sum += float(base_b.view(-1)[0])
        self._weight_norm_sum += float(base_w.norm().item())

    def finalize(self) -> Dict[str, float]:
        count = max(self.total_count, 1)
        batches = max(self.num_batches, 1)
        self.loss_epoch = float(self.total_loss) / float(count)
        self.acc_epoch = float(self.total_correct) / float(count)
        self.grad_norm_avg = float(self.grad_norm_sum) / float(batches)
        self.weight_norm_avg = float(self._weight_norm_sum) / float(batches)
        return {
            "loss": self.loss_epoch,
            "accuracy": self.acc_epoch,
            "grad_norm": self.grad_norm_avg,
            "weight_norm": self.weight_norm_avg,
        }

    def average_weights(self) -> Tensor:
        batches = max(self.num_batches, 1)
        return (self._weight_sum / float(batches)).detach()

    def average_bias(self) -> float:
        batches = max(self.num_batches, 1)
        return float(self._bias_sum) / float(batches)

