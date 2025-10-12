import os
from typing import Iterable

import wandb


def init_wandb(project: str, model_names: Iterable[str]) -> "wandb.sdk.wandb_run.Run":
    """Initialize Weights & Biases with minimal console noise and metric schema.

    - Sets environment variables to reduce console output.
    - Initializes the run and defines per-model step and metrics namespaces.
    """
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")

    run = wandb.init(project=project, settings=wandb.Settings(_disable_stats=True, console="off"))

    for name in model_names:
        wandb.define_metric(f"{name}/step")
        wandb.define_metric(f"{name}/metrics/*", step_metric=f"{name}/step")
        wandb.define_metric(f"{name}/distributions/*", step_metric=f"{name}/step")

    return run
