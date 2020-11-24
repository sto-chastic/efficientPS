import click
from tqdm import tqdm

import torch
from ..dataset.dataset import DataSet
from ..models.full import PSOutput
from ..panoptic_merge.panoptic_merge_arch import panoptic_fusion_module
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

torch.backends.cudnn.benachmark = True

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and ("bn" not in n) and ("se" not in n) and ("fpn.en._fc.weight" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
            else:
                layers.append(n)
                ave_grads.append(0)
                max_grads.append(0)
                print(n, ": nograd")
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.ylim(bottom = -0.001, top=4) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

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
        total_losses = {}

        optimizer.zero_grad()
        loss = 0.0
        for i, loaded_data in enumerate(tqdm(self.loader)):
            losses = {}
            if len(loaded_data.get_bboxes()) == 0:
                continue
            # with torch.autograd.detect_anomaly():
            image = loaded_data.get_image()
            inference = model(image)

            output_image, intermediate_logits = panoptic_fusion_module(
                inference, probe_name="./probe.png"
            )

            loss_fns = loss_class(loaded_data, inference)
            loss = loss + loss_fns.get_total_loss()

            if i % self.minibatch_size == 0 and i > 0:
                loss = loss/self.minibatch_size
                loss.backward()
                loss = 0.0
                # plot_grad_flow(model.named_parameters())
                optimizer.step()
                optimizer.step_scheduler(loss_fns)
                optimizer.zero_grad()
            elif i == len(self.loader):
                div = i - (i // self.minibatch_size) * self.minibatch_size
                loss = loss/div

                loss.backward()
                loss = 0.0
                # plot_grad_flow(model.named_parameters())
                optimizer.step()
                optimizer.step_scheduler(loss_fns)
                optimizer.zero_grad()

            for key, loss_ele in loss_fns.losses_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                if key not in losses:
                    losses[key] = 0.0
                if isinstance(loss_ele, torch.Tensor):
                    total_losses[key] += loss_ele.item()
                    losses[key] += loss_ele.item()
                else:
                    total_losses[key] += loss_ele
                    losses[key] += loss_ele

            print("W Update : Loss={}".format(loss))
            # self.print_loss(loss_fns.losses_dict)

            if i % self.minibatch_size == 0 and i > 0:
                if self.visdom is not None:
                    for loss_type, loss in losses.items():
                        if isinstance(loss, torch.Tensor):
                            loss = loss.item()
                        self.visdom.plot(loss_type, "training", "training", loss/self.minibatch_size)

        for key, loss_ele in total_losses.items():
            total_losses[key] /= i
        return total_losses, inference

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
            for i, loaded_data in enumerate(tqdm(self.loader)):
                if len(loaded_data.get_bboxes()) == 0:
                    continue
                image = loaded_data.get_image()
                inference = model(image)
                loss_fns = loss_class(loaded_data, inference)
                loss = loss_fns.get_total_loss()
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
