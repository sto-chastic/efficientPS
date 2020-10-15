import torch
import torch.nn as nn

from .utilities import DepthSeparableConv2d


class TwoWayFeaturePyramid(nn.Module):
    def __init__(self, activation=nn.GELU):
        super(TwoWayFeaturePyramid, self).__init__()

        self.activation = activation()
        self.block1 = DepthSeparableConv2d(
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )
