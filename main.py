import wandb
from data import DEFAULT_TASK
from benchmarking_csv import update_benchmark_csv
from benchmarking_html import generate_benchmark_html

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
    # Explicitly bind only metrics and distributions under each namespace
    wandb.define_metric("logreg/step")
    wandb.define_metric("self_attention/step")
    wandb.define_metric("attention/step")

    wandb.define_metric("logreg/metrics/*", step_metric="logreg/step")
    wandb.define_metric("self_attention/metrics/*", step_metric="self_attention/step")
    wandb.define_metric("attention/metrics/*", step_metric="attention/step")

    wandb.define_metric("logreg/distributions/*", step_metric="logreg/step")
    wandb.define_metric("self_attention/distributions/*", step_metric="self_attention/step")
    wandb.define_metric("attention/distributions/*", step_metric="attention/step")

    # Unified logger: metrics + optional distributions, per model prefix
    def log_all(model_name: str, epoch: int, metrics: dict, probs, logits):
        merged = {
            f"{model_name}/metrics/loss": metrics.get("loss"),
            f"{model_name}/metrics/accuracy": metrics.get("accuracy"),
            f"{model_name}/metrics/grad_norm": metrics.get("grad_norm"),
            f"{model_name}/metrics/weight_norm": metrics.get("weight_norm"),
            f"{model_name}/step": epoch,
        }
        if probs is not None:
            merged[f"{model_name}/distributions/probabilities"] = wandb.Histogram(probs, num_bins=30)
        if logits is not None:
            merged[f"{model_name}/distributions/logits"] = wandb.Histogram(logits, num_bins=30)
        # Single commit per epoch per model to keep steps tidy
        run.log(merged)

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
        if "self_attention" in run_artifacts:
            generate_run_report(run, run_artifacts["self_attention"], prefix="self_attention")
        if "attention" in run_artifacts:
            generate_run_report(run, run_artifacts["attention"], prefix="attention")
    else:
        generate_run_report(run, run_artifacts)

    # Persist benchmarking to CSV for cross-run comparison
    if isinstance(run_artifacts, dict):
        update_benchmark_csv(task=DEFAULT_TASK, results=run_artifacts, csv_path="benchmarks/benchmarking.csv")

    run.finish()

    # Last step: render a sortable HTML view of the benchmarking CSV
    generate_benchmark_html(csv_path="benchmarks/benchmarking.csv", html_path="benchmarks/index.html")


if __name__ == "__main__":
    main()
