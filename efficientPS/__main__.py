import random

import click
import torch

from .visualizer import LossCurvePlotter
from .models.full import FullModel
from .train_core.optimizer import Optimzer
from .train_core.losses import LossFunctions


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
    help="Root directory of the ground truth. Where the cities folders are",
)
@click.option(
    "-c",
    "--cities_train",
    default=CITIES,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    multiple=True,
    help="The cities list",
)
@click.option(
    "--batches",
    required=False,
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="The number of the batchs for the training",
)
@click.option(
    "--workers",
    required=False,
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="The number of the data-loading workers",
)
@click.option(
    "--use-cuda/--no-cuda",
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
@click.option(
    "--load",
    required=False,
    default=None,
    show_default=True,
    help="Path of a model to load",
)
@click.option(
    "--save-dir",
    required=False,
    default="efficientPS/data",
    show_default=True,
    help="Path of a model to load",
)
@click.option(
    "--epochs",
    required=False,
    default=50000,
    show_default=True,
    type=click.IntRange(min=1),
    help="The number of the data-loading workers",
)
def train_world_models(
    input_dir,
    ground_truth_dir,
    cities,
    batchs,
    workers,
    use_cuda,
    traning_progress_plot,
    load,
    save_dir,
    epochs,
):
    torch.cuda.empty_cache()
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    full_model = FullModel().to(device)

    # This format allows to split the model and train parameters
    # with different optimizers. For now, full model is trained
    optimizer_config = {
        # unique_key: [opt type, parameters, learning rate]
        "full_model": ["adam", full_model.parameters(), 0.07]
    }

    optimizer = Optimzer(optimizer_config)
    if load:
        optimizer.load_state(state_dir=save_dir)
        full_model.load_model(path=save_dir)

    kwargs = {
        "device": device,
        "num_workers": workers,
        "root_folder_inp": input_dir,
        "root_folder_gt": ground_truth_dir,
        "cities_list": cities_train,
        "batches": batches,
    }

    train = Trainer(
        **kwargs,
    )

    if traning_progress_plot:
        from visdom import Visdom

        try:
            vis = Visdom(raise_exceptions=True)
        except ConnectionError:
            RuntimeError(
                "Cannot connect to Visdom.",
                "To visualize: install and run `sudo visdom`, ",
                "or use the --no-traning-progress-plot flag",
            )

        line_plot = LossCurvePlotter(visdom=vis)

    for epoch in range(1, epochs + 1):
        train_loss = train(
            model=full_model,
            loss_func=LossFunctions,
            optimizer=optimizer,
        )

        total_train_loss = train_loss["total_loss"]
        print(f"{epoch}/{epochs} : Loss={total_train_loss}")

        if traning_progress_plot:
            for loss_type, loss in train_loss.items():
                line_plot.add(epoch, loss, loss_type, "training")

        if epoch % 10 == 0:
            full_model.eval()
            full_model.save_model(model_dir)

            optimizer.save_state(state_dir=model_dir)

        


if __name__ == "__main__":
    train_world_models()
