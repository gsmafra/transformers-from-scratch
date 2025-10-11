import wandb

from src.tasks import DEFAULT_TASK
from src.benchmarking_csv import update_benchmark_csv
from src.benchmarking_html import generate_benchmark_html

from src.report import generate_run_report
from src.training import run_training


SEQUENCE_LENGTH = 5
N_SAMPLES = 1000


def main():
    task = DEFAULT_TASK
    project_name = f"transformer-scratchpad-{task}"

    run = wandb.init(
        project=project_name,
        settings=wandb.Settings(_disable_stats=True),  # disable W&B system metrics
    )

    # Configure per-model metrics with minimal duplication
    model_names = ["logreg", "temporal", "self_attention", "attention"]
    for name in model_names:
        wandb.define_metric(f"{name}/step")
        wandb.define_metric(f"{name}/metrics/*", step_metric=f"{name}/step")
        wandb.define_metric(f"{name}/distributions/*", step_metric=f"{name}/step")

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
        task=task,
    )

    # Generate eval/report panels under each model prefix
    if isinstance(run_artifacts, dict):
        for name, artifacts in run_artifacts.items():
            generate_run_report(run, artifacts, prefix=name)
    else:
        generate_run_report(run, run_artifacts)

    # Persist benchmarking to CSV for cross-run comparison
    if isinstance(run_artifacts, dict):
        update_benchmark_csv(task=task, results=run_artifacts, csv_path="benchmarks/benchmarking.csv")

    run.finish()

    # Last step: render a sortable HTML view of the benchmarking CSV
    generate_benchmark_html(csv_path="benchmarks/benchmarking.csv", html_path="benchmarks/index.html")


if __name__ == "__main__":
    main()

