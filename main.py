import wandb

from report import generate_run_report
from training import run_training


SEQUENCE_LENGTH = 5
N_SAMPLES = 1000


def main():
    run = wandb.init(
        project="transformer-scratchpad",
        settings=wandb.Settings(_disable_stats=True),  # disable W&B system metrics
    )

    # Use model-specific step metrics to avoid global step collisions
    wandb.define_metric("logreg/*", step_metric="logreg/step")
    wandb.define_metric("temporal/*", step_metric="temporal/step")

    # Unified logger: metrics + optional distributions, per model prefix
    def log_all(model_name: str, epoch: int, metrics: dict, probs, logits):
        to_log = {
            f"{model_name}/metrics/loss": metrics.get("loss"),
            f"{model_name}/metrics/accuracy": metrics.get("accuracy"),
            f"{model_name}/metrics/grad_norm": metrics.get("grad_norm"),
            f"{model_name}/metrics/weight_norm": metrics.get("weight_norm"),
            f"{model_name}/metrics/bias_abs": metrics.get("bias_abs"),
            f"{model_name}/step": epoch,
        }
        run.log(to_log)

        if probs is not None:
            run.log({
                f"{model_name}/distributions/probabilities": wandb.Histogram(probs, num_bins=30),
                f"{model_name}/step": epoch,
            })
        if logits is not None:
            run.log({
                f"{model_name}/distributions/logits": wandb.Histogram(logits, num_bins=30),
                f"{model_name}/step": epoch,
            })

    run_artifacts = run_training(
        sequence_length=SEQUENCE_LENGTH,
        n_samples=N_SAMPLES,
        seed=0,
        on_log=log_all,
        hist_every=10,
    )

    # Generate eval/report panels for both models under separate prefixes
    if isinstance(run_artifacts, dict):
        if "logreg" in run_artifacts:
            generate_run_report(run, run_artifacts["logreg"], prefix="logreg")
        if "temporal" in run_artifacts:
            generate_run_report(run, run_artifacts["temporal"], prefix="temporal")
    else:
        generate_run_report(run, run_artifacts)

    run.finish()


if __name__ == "__main__":
    main()
