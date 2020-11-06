import torch
from torch.optim import SGD, Adam, AdamW


class Optimizer:
    def __init__(
        self,
        optimizer_config,
        min_lr=0.00001,
        patience=40,
        milestones=[32000, 44000],
        gamma=0.1,
    ):
        self.filename = "optimizer.pt"

        def select_optimizer(op_type, parameters, lr):
            if op_type == "adam":
                return AdamW(parameters, lr=lr)
            elif op_type == "sgd":
                return SGD(parameters, lr=lr)

        def select_scheduler(op_type, optimizer):
            if op_type == "adam":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    "min",
                    min_lr=min_lr,
                    patience=patience,
                )
            elif op_type == "sgd":
                return torch.optim.MultiStepLR(
                    optimizer, milestones=milestones, gamma=gamma
                )

        self.optimizers = {
            key: select_optimizer(*type_params_lr)
            for key, type_params_lr in optimizer_config.items()
        }
        self.schedulers = {
            key: select_scheduler(type_params_lr[0], self.optimizers[key])
            for key, type_params_lr in optimizer_config.items()
        }

    def zero_grad(self):
        [opt.zero_grad() for _, opt in self.optimizers.items()]

    def step(self):
        [opt.step() for _, opt in self.optimizers.items()]

    def step_scheduler(self, loss):
        [sch.step(loss[key]) for key, sch in self.schedulers.items()]

    def load_state(self, state_dir) -> None:
        if not state_dir.endswith("/"):
            state_dir = state_dir + "/"
        state = torch.load(f"{state_dir}{self.filename}")
        for key, opt in self.optimizer.items():
            opt.load_state_dict(state[key + "_optimizer"])
        for key, sch in self.scheduler.items():
            sch.load_state_dict(state[key + "_scheduler"])

    def save_state(self, state_dir) -> None:
        if not state_dir.endswith("/"):
            state_dir = state_dir + "/"
        state = dict()
        for key, opt in self.optimizer.items():
            state[key + "_optimizer"] = opt.state_dict()
        for key, sch in self.scheduler.items():
            state[key + "_scheduler"] = sch.state_dict()
        torch.save(state, f"{state_dir}{self.filename}")
