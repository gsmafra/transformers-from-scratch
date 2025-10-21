from src.benchmarking.csv import update_benchmark_csv
from src.benchmarking.html import generate_benchmark_html
from src.reporting.report import generate_run_report
from src.training import run_training
from src.reporting.wandb_init import init_wandb


N_SAMPLES = 1000
TASK = "sign_of_winner"


def main():
    task = TASK
    project_name = f"transformer-scratchpad-{task}"

    # Configure W&B and per-model metric namespaces
    model_names = [
        "logreg",
        "mlp",
        "temporal",
        "self_attention",
        "self_attention_qkv",
        "self_attention_qkv_pos",
        "attention",
    ]
    run = init_wandb(project=project_name, model_names=model_names)

    run_artifacts = run_training(
        n_samples=N_SAMPLES,
        seed=0,
        run=run,
        task=task,
        model_names=model_names,
    )

    # Generate eval/report panels under each model prefix
    for name, artifacts in run_artifacts.items():
        generate_run_report(run, artifacts, prefix=name)

    run.finish()

    update_benchmark_csv(task=task, results=run_artifacts, csv_path="benchmarks/benchmarking.csv")
    generate_benchmark_html(csv_path="benchmarks/benchmarking.csv", html_path="benchmarks/index.html")


if __name__ == "__main__":
    main()
