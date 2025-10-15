from src.benchmarking.csv import update_benchmark_csv
from src.benchmarking.html import generate_benchmark_html
from src.report import generate_run_report
from src.tasks import DEFAULT_TASK
from src.training import run_training
from src.wandb_init import init_wandb


SEQUENCE_LENGTH = 5
N_SAMPLES = 1000


def main():
    task = DEFAULT_TASK
    project_name = f"transformer-scratchpad-{task}"

    # Configure W&B and per-model metric namespaces
    model_names = ["logreg", "temporal", "self_attention", "self_attention_qkv", "attention"]
    run = init_wandb(project=project_name, model_names=model_names)

    run_artifacts = run_training(
        sequence_length=SEQUENCE_LENGTH,
        n_samples=N_SAMPLES,
        seed=0,
        run=run,
        task=task,
    )

    # Generate eval/report panels under each model prefix
    for name, artifacts in run_artifacts.items():
        generate_run_report(run, artifacts, prefix=name)

    run.finish()

    update_benchmark_csv(task=task, results=run_artifacts, csv_path="benchmarks/benchmarking.csv")
    generate_benchmark_html(csv_path="benchmarks/benchmarking.csv", html_path="benchmarks/index.html")


if __name__ == "__main__":
    main()
