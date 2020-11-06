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
        crop=(1024, 2048),
        batches=1,
        num_workers=1,
        train: bool = True,
        visdom=None,
    ):
        self.dataset = DataSet(
            root_folder_inp=root_folder_inp,
            root_folder_gt=root_folder_gt,
            cities_list=cities_list,
            device=device,
            crop=crop,
        )
        self.batches = batches
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batches,
            shuffle=train,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        self.device = device
        self.train = train

        self.visdom = visdom

    @staticmethod
    def collate_fn(batch):
        # TODO(David): Only works with batch size 1
        # should write my own collate fn for the DataLoader
        # and extend my PSSamples functions to multiple samples
        if len(batch) == 1:
            return batch[0]
        else:
            raise NotImplementedError()


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
        loss_class,
        optimizer,
    ):
        model.train()
        losses = {}

        for loaded_data in self.loader:
            image = loaded_data.get_image()
            optimizer.zero_grad()
            inference = model(image)
            loss_fns = loss_class(loaded_data, inference)
            loss = loss_fns.get_total_loss()

            loss.backward()

            optimizer.step()
            for key, _ in loss_fns.losses_dict.items():
                if key not in losses:
                    losses[key] = 0.0
                losses[key] += loss_fns.losses_dict[key].item()

            optimizer.step_scheduler(loss_fns)
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
        loss_class,
        optimizer,
    ):
        model.eval()
        losses = {}
        with torch.no_grad():
            for loaded_data in self.loader:
                image = loaded_data.get_image()
                inference = model(image)
                loss_fns = loss_class(loaded_data, inference)
                loss = loss_fns.get_total_loss()

                for key, _ in loss.items():
                    if key not in losses:
                        losses[key] = 0.0
                    losses[key] += loss[key].item()
        return losses


if __name__ == "__main__":
    run()
