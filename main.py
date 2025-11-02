from src.benchmarking.csv import update_benchmark_csv
from src.benchmarking.overlap import update_overlap_csv
from src.benchmarking.html import generate_benchmark_html
from src.reporting.report import generate_run_report
from src.training import run_training
from src.reporting.wandb_init import init_wandb
from tqdm import tqdm
from src.tasks import prepare_data


N_SAMPLES = 1000
# Configure which tasks and models to run
TASKS = [
    "sign_of_winner",
    "sign_of_second_place",
    "has_pos_and_neg",
    "has_all_tokens",
    "any_abs_gt_one",
    "single_digit_string_sum",
    "single_digit_string_sum_swapped",
    "multi_digit_sum",
]
MODELS = [
    "logreg",
    "mlp",
    "temporal",
    "self_attention",
    "singlelayer_transformer",
    "bahdanau_attention",
    "multilayer_transformer",
]


def main():
    # Run all configured tasks × models
    for task in tqdm(TASKS, desc="tasks"):
        project_name = f"transformer-scratchpad-{task}"
        run = init_wandb(project=project_name, model_names=MODELS)

        run_artifacts = run_training(
            n_samples=N_SAMPLES,
            seed=0,
            run=run,
            task=task,
            model_names=MODELS,
        )

        # Generate eval/report panels under each model prefix
        for name, artifacts in run_artifacts.items():
            generate_run_report(run, artifacts, prefix=name)

        run.finish()

        update_benchmark_csv(task=task, results=run_artifacts, csv_path="benchmarks/benchmarking.csv")
        # Compute and record train-test overlap for this task using the same seeds
        x_tr, _ = prepare_data(n_samples=N_SAMPLES, seed=0, task=task)
        x_te, _ = prepare_data(n_samples=N_SAMPLES, seed=1, task=task)
        update_overlap_csv(task=task, x=x_tr, x_test=x_te, csv_path="benchmarks/overlap.csv")

    # Refresh comparison dashboard after all tasks are processed
    generate_benchmark_html(csv_path="benchmarks/benchmarking.csv", html_path="benchmarks/index.html", overlap_csv_path="benchmarks/overlap.csv")


if __name__ == "__main__":
    main()
