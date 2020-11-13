import random

import click
import torch

# from .visualizer import LossCurvePlotter
from .models.full import FullModel
from .train_core.optimizer import Optimizer
from .train_core.losses import LossFunctions
from . import *
from .dataset import *
from .models import *
from .train_core.optimizer import Optimizer
from .train_core.trainer import Trainer, Validator
from .train_core.losses import LossFunctions
from .panoptic_merge.panoptic_merge_arch import panoptic_fusion_module


@click.command()
@click.option(
    "-i",
    "--input-dir",
    default=INPUT_DIR,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of the images. Where the cities folders are",
)
@click.option(
    "-i",
    "--input-val",
    default=INPUT_VAL,
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
    "-gt",
    "--ground-truth-val-dir",
    default=GT_VAL_DIR,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of the ground truth. Where the cities folders are",
)
@click.option(
    "-c",
    "--cities_train",
    default=CITIES,
    show_default=True,
    multiple=True,
    help="The cities list",
)
@click.option(
    "-c",
    "--cities_validate",
    default=CITIES_VAL,
    show_default=True,
    multiple=True,
    help="The cities list",
)
@click.option(
    "--batches",
    required=False,
    default=1,
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
@click.option(
    "--crop-sizes",
    default=[1024, 2048],
    show_default=True,
    multiple=True,
    type=click.IntRange(min=1, max=2048),
    help="Enter 2 values: Crop the images at random locations to be of Height, Widht",
)
@click.option(
    "--nms-threshold",
    default=0.4,
    show_default=True,
    type=click.FloatRange(min=0.1, max=0.99),
    help="NMS threshold",
)
@click.option(
    "--minibatch-size",
    default=4,
    show_default=True,
    type=click.IntRange(min=1),
    help="NMS threshold",
)
def train_ps(
    input_dir,
    input_val,
    ground_truth_dir,
    ground_truth_val_dir,
    cities_train,
    cities_validate,
    batches,
    workers,
    use_cuda,
    traning_progress_plot,
    load,
    save_dir,
    epochs,
    crop_sizes,
    nms_threshold,
    minibatch_size,
):
    torch.cuda.empty_cache()
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    full_model = FullModel(
        len(THINGS), len(STUFF), ANCHORS.to(device), nms_threshold
    ).to(device)

    # This format allows to split the model and train parameters
    # with different optimizers. For now, full model is trained

    # NOTE: I selected to use AdamW as it is a robust optimizer
    # and requires less tuning than the proposed SGD (no warmup)
    # see https://www.fast.ai/2018/07/02/adam-weight-decay/
    optimizer_config = {
        # unique_key: [opt type, parameters, learning rate]
        "full_model": ["adamw", full_model.parameters(), 0.07]
    }

    optimizer = Optimizer(optimizer_config)
    if load:
        optimizer.load_state(state_dir=save_dir)
        full_model.load_model(path=save_dir)

    line_plotter = None
    if traning_progress_plot:
        from .extras.plotter import VisdomLinePlotter

        try:
            line_plotter = VisdomLinePlotter()

        except ConnectionError:
            RuntimeError(
                "Cannot connect to Visdom.",
                "To visualize: install and run `sudo visdom`, ",
                "or use the --no-traning-progress-plot flag",
            )

    kwargs = {
        "device": device,
        "num_workers": workers,
        "root_folder_inp": input_dir,
        "root_folder_gt": ground_truth_dir,
        "cities_list": cities_train,
        "batches": batches,
        "crop": crop_sizes,
        "visdom": line_plotter,
    }

    train = Trainer(minibatch_size, **kwargs)

    kwargs["root_folder_inp"] = input_val
    kwargs["root_folder_gt"] = ground_truth_val_dir
    kwargs["cities_list"] = cities_validate

    validate = Validator(**kwargs)

    current_best_loss = 1e7
    for epoch in range(1, epochs + 1):
        train(
            model=full_model,
            loss_class=LossFunctions,
            optimizer=optimizer,
        )
        validation_loss, inference = validate(
            model=full_model,
            loss_class=LossFunctions,
            optimizer=optimizer,
        )

        total_validation_loss = validation_loss["full_model"]
        print(f"{epoch}/{epochs} : Loss={total_validation_loss}")

        output_image, intermediate_logits = panoptic_fusion_module(
            inference, probe_name="./validation{}.png".format(epoch)
        )

        if line_plotter:
            for loss_type, loss in train_loss.items():
                line_plotter.plot(loss_type, "validation", loss)

        if total_validation_loss <= current_best_loss:
            current_best_loss = total_validation_loss
            full_model.eval()
            full_model.save_model(save_dir)

            optimizer.save_state(state_dir=save_dir)


if __name__ == "__main__":
    train_ps()
