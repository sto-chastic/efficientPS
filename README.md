# efficientPS

Code provided for the paper [EfficientPS: Efficient Panoptic Segmentation](https://arxiv.org/pdf/2004.02307.pdf) by Rohit et al.

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
