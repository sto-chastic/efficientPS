import random

import click
import numpy as np
import torch

from .visualizer import LossCurvePlotter
from .functor import Trainer
from .model import WorldModelLoss, WorldModels, WorldModelOptimizer


@click.command()
@click.option(
    "-i",
    "--input-dir",
    default=INPUT_DIR,
    multiple=True,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of the images. Where the cities folders are",
)
@click.option(
    "-gt",
    "--ground-truth-dir",
    default=GT_DIR,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory to save the updated model.",
)
@click.option(
    "-gt",
    "--ground-truth-dir",
    default=GT_DIR,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory to save the updated model.",
)
@click.option(
    "-b",
    "--base-model-dir",
    required=False,
    type=click.Path(exists=True, file_okay=True),
    help="The initial model directory path if exists",
)
@click.option(
    "-v",
    "--vectormap-dir",
    default=MAP_DATA_DIR,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="The vectormap directory",
)
@click.option(
    "--number-training-points",
    required=False,
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="The number of training points. If 0, it will train for 1 epoch",
)
@click.option(
    "--batchs",
    required=False,
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="The number of the batchs for the training",
)
@click.option(
    "--data-workers",
    required=False,
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="The number of the data loading workers",
)
@click.option(
    "--use-cuda/--no-cuda",
    required=False,
    default=True,
    show_default=True,
    help="Whether use cuda or not",
)
@click.option(
    "--release/--debug",
    required=False,
    default=True,
    show_default=True,
    help="Whether use cuda or not",
)
@click.option(
    "--traning-progress-plot/--no-traning-progress-plot",
    required=False,
    default=True,
    show_default=True,
    help="Show the small training plots with the epoch and status",
)
def train_world_models(
    episodes_dirs,
    model_dir,
    base_model_dir,
    vectormap_dir,
    number_training_points,
    batchs,
    data_workers,
    use_cuda,
    release,
    traning_progress_plot,
):
    torch.cuda.empty_cache()
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    world_models = WorldModels().to(device)
    if base_model_dir:
        world_models.load_weight(weight_dir=base_model_dir)

    learning_rates = {
        "ego": 6e-4,
        "map": 0.001,
        "obstacles": 0.001,
        "reward": 0.001,
    }

    optimizer = WorldModelOptimizer(world_models, learning_rates)
    if base_model_dir:
        optimizer.load_state(state_dir=base_model_dir)

    episode_files = find_episode_files_in_dirs(episodes_dirs, recursive=True)
    if number_training_points:
        epochs = number_training_points // len(episode_files) + 1
    else:
        epochs = 1
    print(
        f"Found {len(episode_files)} episode files. Will do {epochs} epochs."
    )

    refiner_config = RawObservationRefinerConfig()

    kwargs = {
        "device": device,
        "num_workers": data_workers,
        "refiner_config": refiner_config,
        "prediction_steps": prediction_steps,
    }
    if not release:
        kwargs["num_workers"] = 0

    train = Trainer(
        episode_files=episode_files,
        vectormap_dir=vectormap_dir,
        batchs=batchs,
        **kwargs,
    )

    if traning_progress_plot:
        from visdom import Visdom
        try:
            vis = Visdom(raise_exceptions=True)
        except ConnectionError:
            raise RuntimeError(
                "Cannot connect to Visdom.",
                "Maybe forgot to launch a visdom?",
                "Please `sudo visdom`.",
                "or pass --no-traning-progress-plot",
            )

        line_plot = LossCurvePlotter(visdom=vis)

    for epoch in range(1, epochs + 1):
        train_loss = train(
            model=world_models,
            loss_func=WorldModelLoss(),
            optimizer=optimizer,
        )
        optimizer.step_scheduler(train_loss)

        total_train_loss = train_loss["total_loss"]
        print(f"{epoch}/{epochs} : Loss={total_train_loss}")

        train.visualize(model=world_models, loss_func=WorldModelLoss())
        if traning_progress_plot:
            for loss_type, loss in train_loss.items():
                line_plot.add(epoch, loss, loss_type, "training")

    world_models.eval()  # ROADX-2242
    world_models.save_weight(weight_dir=model_dir)
    world_models.save_jit_weight(weight_dir=model_dir)

    optimizer.save_state(state_dir=model_dir)


if __name__ == "__main__":
    train_world_models()
