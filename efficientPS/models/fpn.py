import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import (
    DepthSeparableConv2d,
    MobileInvertedBottleneck,
    conv_1x1_bn,
)


class TwoWayFeaturePyramid(nn.Module):
    def __init__(self, activation=nn.LeakyReLU):
        super(TwoWayFeaturePyramid, self).__init__()

        # Main branch

        self.block1 = MobileInvertedBottleneck(
            in_channels=3,
            out_channels=48,
            stride=2,
            expand_ratio=3,
        )  # output 48, 512, 1024
        self.block2 = MobileInvertedBottleneck(
            in_channels=48,
            out_channels=24,
            stride=1,
            expand_ratio=3,
        )  # output 24, 512, 1024
        self.block3 = MobileInvertedBottleneck(
            in_channels=24,
            out_channels=40,
            stride=2,
            expand_ratio=3,
        )  # output 40, 256, 512
        self.block4 = MobileInvertedBottleneck(
            in_channels=40,
            out_channels=64,
            stride=2,
            expand_ratio=3,
        )  # output 64, 128, 256
        self.block5 = MobileInvertedBottleneck(
            in_channels=64,
            out_channels=128,
            stride=2,
            expand_ratio=3,
        )  # output 128, 64, 128
        self.block6 = MobileInvertedBottleneck(
            in_channels=128,
            out_channels=176,
            stride=1,
            expand_ratio=3,
        )  # output 176, 64, 128
        self.block7 = MobileInvertedBottleneck(
            in_channels=176,
            out_channels=304,
            stride=2,
            expand_ratio=3,
        )  # output 304, 32, 64
        self.block8 = MobileInvertedBottleneck(
            in_channels=304,
            out_channels=512,
            stride=1,
            expand_ratio=3,
        )  # output 512, 32, 64
        self.block9 = MobileInvertedBottleneck(
            in_channels=512,
            out_channels=2048,
            stride=1,
            expand_ratio=3,
        )  # output 2048, 32, 64

        # Bottom-up branch
        self.times4_reduction_bu = conv_1x1_bn(40, 256)
        self.downsample4 = nn.AdaptiveAvgPool2d((128, 256))
        self.times8_reduction_bu = conv_1x1_bn(64, 256)
        self.downsample8 = nn.AdaptiveAvgPool2d((64, 128))
        self.times16_reduction_bu = conv_1x1_bn(176, 256)
        self.downsample16 = nn.AdaptiveAvgPool2d((32, 64))
        self.times32_reduction_bu = conv_1x1_bn(2048, 256)

        # Top-down branch
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.times4_reduction_td = conv_1x1_bn(40, 256)
        self.times8_reduction_td = conv_1x1_bn(64, 256)
        self.times16_reduction_td = conv_1x1_bn(176, 256)
        self.times32_reduction_td = conv_1x1_bn(2048, 256)

        # Ps Separable Convolutions
        self.p32conv = DepthSeparableConv2d(256, 256)
        self.p16conv = DepthSeparableConv2d(256, 256)
        self.p8conv = DepthSeparableConv2d(256, 256)
        self.p4conv = DepthSeparableConv2d(256, 256)

    def forward(self, inp):
        # Main and bottom-up
        x = self.block1(inp)
        x = self.block2(x)
        x = self.block3(x)

        x_bu_1 = self.times4_reduction_bu(x)
        x_td_4_ = self.times4_reduction_td(x)

        x = self.block4(x)

        x_bu_2 = self.times8_reduction_bu(x) + self.downsample4(x_bu_1)
        x_td_3_ = self.times8_reduction_td(x)

        x = self.block5(x)
        x = self.block6(x)

        x_bu_3 = self.times16_reduction_bu(x) + self.downsample8(x_bu_2)
        x_td_2_ = self.times16_reduction_td(x)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x_bu_4 = self.times32_reduction_bu(x) + self.downsample16(x_bu_3)
        x_td_1 = self.times32_reduction_td(x)

        # Top-down branch computation
        x_td_2 = x_td_2_ + self.upsample(x_td_1)
        x_td_3 = x_td_3_ + self.upsample(x_td_2)
        x_td_4 = x_td_4_ + self.upsample(x_td_3)

        # Final Ps
        p32 = self.p32conv(x_td_1 + x_bu_4)
        p16 = self.p16conv(x_td_2 + x_bu_3)
        p8 = self.p8conv(x_td_3 + x_bu_2)
        p4 = self.p4conv(x_td_4 + x_bu_1)

        return p32, p16, p8, p4


if __name__ == "__main__":
    fpn = TwoWayFeaturePyramid().cuda()
    p32, p16, p8, p4 = fpn(torch.rand(3, 3, 1024, 2048))
    print("p32", p32.shape)
    print("p16", p16.shape)
    print("p8", p8.shape)
    print("p4", p4.shape)
