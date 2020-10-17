import torch
import torch.nn as nn

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*padding) / stride) + 1
    return output

class DepthSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
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


# def conv_3x3_bn(in_channels, out_channels, stride):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU6(inplace=True)
#     )


def conv_1x1_bn(in_channels, out_channels, activation=nn.LeakyReLU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class MobileInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, activation=nn.ReLU6):
        super(MobileInvertedBottleneck, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(in_channels * expand_ratio)
        self.residual_shortcut = (self.stride == 1 and in_channels == out_channels)

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)