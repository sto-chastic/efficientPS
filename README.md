# efficientPS

Code provided for the paper [EfficientPS: Efficient Panoptic Segmentation](https://arxiv.org/pdf/2004.02307.pdf) by Rohit et al.

My explanation of the paper can be found in the following presention [here](https://docs.google.com/presentation/d/1eRmAbOkiaD6nZz-zaJMASPL7F_pegBrerSufvLhw0_Q/edit?usp=sharing)

The network is designed to use Gradient Checkpointing due to its large size. As explained [here](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb) it works by dividing the model or parts of the model in various segments and executing the segments without taping them in the forward pass i.e. their taping is delayed until the backward pass. This results in not storing the entire model in memory but only segments, reducing the memory usage at the expense of computation speed during training.

Current preliminary state.

## Setup

To get the submodules, run: 

```bash
git submodule update --recursive --init
```

It will get the required submodules.

### Conda Environment

I provide a yml [file](environment.yml) that you can use to create a conda environment for running this code.

### VSCode
I provide a .vscode folder that can be used to run the code using VSCODE.

## Train

For training run:

```bash
python3 -m efficientPS -i path/to/train/inputs -gt path/to/gt/train -c aachen -c bochum --no-traning-progress-plot --crop-sizes 512 --crop-sizes 1024 --minibatch-size 1 --traning-progress-plot
```

For more options run

```bash
python3 -m efficientPS --help
```

## Visdom

Visdom allows you to monitor in real time the training progress, in a separate terminal run:
```bash
visdom
```
or 
```bash
python3 -m visdom.server
```

then navigate to http://localhost:8097 in your browser
