import click

import torch
from ..dataset.dataset import DataSet
from ..models.full import PSOutput


class Core:
    def __init__(
        self,
        root_folder_inp,
        root_folder_gt,
        cities_list,
        device,
        batches=1,
        num_workers=1,
        train: bool = True,
        visdom=None,
    ):
        self.dataset = DataSet(
            root_folder_inp=root_folder_inp,
            root_folder_gt=root_folder_gt,
            cities_list=cities_list,
        )
        self.batches = batches
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batches,
            shuffle=train,
            num_workers=num_workers,
            drop_last=True,
        )
        self.device = device
        self.train = train

        self.visdom = visdom


class Trainer(Core):
    def __init__(
        self,
        **kargs,
    ):
        super().__init__(
            train=True,
            **kargs,
        )

    def __call__(
        self,
        model,
        loss_fn,
        optimizer,
    ):
        model.train()
        losses = {}

        for loaded_data in self.loader:
            with torch.autograd.detect_anomaly():
                # TODO(David): Only works with batch size 1
                # should write my own collate fn for the DataLoader
                # and extend my PSSamples functions to multiple samples
                image = loaded_data.get_image()
                optimizer.zero_grad()
                inference = model(image)
                loss = loss_fn(inference, loaded_data)

                loss["total_loss"].backward()

                optimizer.step()
                for key, _ in loss.items():
                    if key not in losses:
                        losses[key] = 0.0
                    losses[key] += loss[key].item()

                optimizer.step_scheduler(losses)
        return losses


class Validator(Core):
    def __init__(
        self,
        **kargs,
    ):
        super().__init__(
            train=False,
            **kargs,
        )

    def __call__(
        self,
        model,
        loss_fn,
        optimizer,
    ):
        model.eval()
        losses = {}
        with torch.no_grad():
            for loaded_data in self.loader:
                image = loaded_data.get_image()
                inference = model(image)
                loss = loss_fn(inference, loaded_data)

                for key, _ in loss.items():
                    if key not in losses:
                        losses[key] = 0.0
                    losses[key] += loss[key].item()
        return losses


if __name__ == "__main__":
    run()
