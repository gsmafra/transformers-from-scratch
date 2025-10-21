Progressive Transformers Scratchpad

- Purpose: train simple-to-expressive sequence models on small synthetic tasks, log results, and generate lightweight reports.
- Scope: tasks include numeric feature tasks and tokenized arithmetic string tasks; models are pluggable and may change over time.

How It Works

- Data and tasks
  - Tasks live under `src/tasks/` and subclass `Task` (`src/tasks/base.py`). Each task defines a `feature_dim` and a `sequence_length`.
  - The selected task is defined in `main.py` via `TASK`. `prepare_data(n_samples, seed, task=...)` uses that to generate data (train uses `seed`, test uses `seed+1`). Sequence length is derived from the selected task.
- Models
  - Models live under `src/models/` and expose lightweight “access” wrappers by subclassing `ModelAccess`.
  - The set of models trained is determined in `src/models/registry.py` by `build_models(...)`. Inspect that file for the current selection.
  - Training epochs, learning rates, and mini-batch size are defined per model via the `ModelAccess` constructor.
- Training
  - `src/training.py` prepares data once, builds the models, and trains each model for its configured epochs.
  - Per-epoch, it logs aggregated train metrics and also evaluates the model on the test set.

Logging and Reports

- Weights & Biases (W&B)
  - Initialized via `src/reporting/wandb_init.py`. Each model logs under a namespaced prefix (the model’s name).
  - Logged every epoch per model:
    - Train histories: `metrics/loss` (line plot), `metrics/accuracy_train_history` (line plot).
    - Test histories: `metrics/loss_test_history` (line plot), `metrics/accuracy_test_history` (line plot).
    - Scalars: `metrics/loss_train`, `metrics/accuracy_train`, `metrics/loss_test`, `metrics/accuracy_test`, plus any model-specific extras.
  - End-of-run panels per model for both splits: ROC curve, PR curve, and confusion matrix.
- HTML artifacts
  - Per-model readable report is written to `artifacts/<task>/<model>.html` with parameter tables and misclassified examples for both train and test.
  - The comparison dashboard is written to `benchmarks/index.html`. It shows two tables (Train and Test) built from `benchmarks/benchmarking.csv`.

Running

- Requirements: Python 3.8+, Torch, NumPy, W&B.
- Authentication: run `wandb login` once or set `WANDB_API_KEY`.
- Start a run: `python main.py`.
- Offline mode: `WANDB_MODE=offline python main.py` (sync later with `wandb sync`).

Configuration (stable touchpoints)

- Task: set `TASK` in `main.py`.
- Sample size: edit `N_SAMPLES` in `main.py`.
- Models included: edit `build_models(...)` in `src/models/registry.py`.
- Model hyperparameters: adjust the respective `ModelAccess` subclass (epochs, LR, batch size).
- Sequence length and feature dimension: come from the selected task; no need to set them in `main.py`.

Extending

- Add a task: create a new subclass of `Task` in `src/tasks/`, implement `label(...)` and `generate_candidates(...)`, set `sequence_length`, and register it in `src/tasks/registry.py`.
- Add a model: add a new `ModelAccess` subclass in `src/models/` and include it in `build_models(...)`.
