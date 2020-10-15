import click

import torch


class Core:
    def __init__(
        self,
        files_dir,
        device: torch.device,
        batchs: int = 1,
        num_workers: int = 1,
        train: bool = True,
        visdom=None,
    ):
        self.dataset = DataSet(
            refiner_config=refiner_config,
            episode_list=episode_files,
            vectormap_dir=vectormap_dir,
            prediction_steps=prediction_steps,
        )
        self.batchs = batchs
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batchs,
            shuffle=train,
            num_workers=num_workers,
            drop_last=True,
        )
        self.device = device
        self.train = train
        self.prediction_steps = prediction_steps
        self.start_forecast = start_forcast_step
        if self.prediction_steps <= self.start_forecast:
            raise ValueError(
                "start_forecast_step should be smaller than prediction_step"
            )

        if visdom is None:
            try:
                visdom = Visdom(raise_exceptions=True)
            except ConnectionError:
                print("Cannot connect to Visdom")
        self.visdom = visdom


class Trainer(Core):
    def __init__(
        self, files_dir, **kargs,
    ):
        super().__init__(
            files_dir=files_dir, train=True, **kargs,
        )
        self.gradient_clip = gradient_clip

    def __call__(
        self,
        model: WorldModels,
        loss_func: WorldModelLoss,
        optimizer: WorldModelOptimizer,
    ) -> Dict[str, float]:
        model.train()
        losses: Dict[str, float] = {}

        for loaded_data in self.loader:
            with torch.autograd.detect_anomaly():
                observations, actions, truth = self.extract(loaded_data)
                optimizer.zero_grad()
                inference = model.inference(observations, actions)
                loss = loss_func(inference, truth)
                loss["total_loss"].backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clip
                )
                optimizer.step()
                for key, _ in loss.items():
                    if key not in losses:
                        losses[key] = 0.0
                    losses[key] += loss[key].item()
        return losses


class Validator(Core):
    def __init__(
        self, files_dir: List[str], vectormap_dir: str, **kargs,
    ):
        super().__init__(
            files_dir=files_dir,
            vectormap_dir=vectormap_dir,
            train=False,
            **kargs,
        )

    def __call__(
        self, model: WorldModels, loss_func: WorldModelLoss,
    ) -> Dict[str, float]:
        model.eval()
        losses: Dict[str, float] = {}
        with torch.no_grad():
            for loaded_data in self.loader:
                observations, actions, truth = self.extract(loaded_data)
                inference = model.inference(observations, actions)
                loss = loss_func(inference, truth)
                for key, _ in loss.items():
                    if key not in losses:
                        losses[key] = 0.0
                    losses[key] += loss[key].item()
        return losses


if __name__ == "__main__":
    run()
