import click
from tqdm import tqdm

import torch
from ..dataset.dataset import DataSet
from ..models.full import PSOutput
from ..panoptic_merge.panoptic_merge_arch import panoptic_fusion_module

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
        minibatch_size,
        **kargs,
    ):
        super().__init__(
            train=True,
            **kargs,
        )
        self.minibatch_size = minibatch_size
    def __call__(
        self,
        model,
        loss_class,
        optimizer,
    ):
        model.train()
        losses = {}

        optimizer.zero_grad()
        for i, loaded_data in enumerate(tqdm(self.loader)):
            total_loss = 0.0
            image = loaded_data.get_image()

            inference = model(image)
            loss_fns = loss_class(loaded_data, inference)
            loss = loss_fns.get_total_loss()

            total_loss += loss

            for key, loss_ele in loss_fns.losses_dict.items():
                if key not in losses:
                    losses[key] = 0.0
                if isinstance(loss_ele, torch.Tensor):
                    losses[key] += loss_ele.item()
                else:
                    losses[key] += loss_ele

            if i % self.minibatch_size == 0 and i > 0:
                total_loss /= self.minibatch_size
                total_loss.backward()

                optimizer.step()
                optimizer.step_scheduler(loss_fns)
                optimizer.zero_grad()
                print("W Update : Loss={}".format(total_loss))
                self.print_loss(loss_fns.losses_dict)
                panoptic_fusion_module(inference)
            
            elif i == len(self.loader):
                div = i - (i // self.minibatch_size)*self.minibatch_size
                total_loss /= div
                total_loss.backward()

                optimizer.step()
                optimizer.step_scheduler(loss_fns)
                optimizer.zero_grad()
                print("W Update : Loss={}".format(total_loss))
                panoptic_fusion_module(inference, probe_name="./probe.png")

            if self.visdom is not None:
                for loss_type, loss in loss_fns.losses_dict.items():
                    if isinstance(loss, torch.Tensor):
                        loss =  loss.item()
                    self.visdom.plot(loss_type, "training", "training", loss)

        for key, loss_ele in losses.items():
            losses[key] /= i
        return losses, inference

    @staticmethod
    def print_loss(losses_dict):
        for key, loss_ele in losses_dict.items():
            print("{}: {}".format(key, loss_ele))


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
                for key, loss_ele in loss_fns.losses_dict.items():
                    if key not in losses:
                        losses[key] = 0.0
                    if isinstance(loss_ele, torch.Tensor):
                        losses[key] += loss_ele.item()
                    else:
                        losses[key] += loss_ele

        for key, loss_ele in losses.items():
            losses[key] /= i
        return losses, inference


if __name__ == "__main__":
    run()
