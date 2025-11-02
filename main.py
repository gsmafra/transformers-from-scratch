from src.training import run_all_tasks_and_reports


N_SAMPLES = 1000
# Configure which tasks and models to run
TASKS = [
    "sign_of_winner",
    "sign_of_second_place",
    "has_pos_and_neg",
    "has_all_tokens",
    "any_abs_gt_one",
    "single_digit_sum",
    "single_digit_sum_swapped",
    "aa+bb=cc",
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
    run_all_tasks_and_reports(
        n_samples=N_SAMPLES,
        seed=0,
        tasks=TASKS,
        model_names=MODELS,
    )


if __name__ == "__main__":
    main()
