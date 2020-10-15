import torch
import torch.nn as nn


class DepthSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(DepthSeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.depth_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.depth_conv(x)
        return x
