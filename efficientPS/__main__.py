import click

import torch
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from .train import Train
from .utils import Buffer, ProgressPlotter, apply, create_new_dir, to_tensor


@click.command()
@click.option(
    "--train/--test",
    required=False,
    default=true,
    show_default=True,
    help="Train or Test mode",
)
def run(train):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    trainer = Trainer()


if __name__ == "__main__":
    run()
