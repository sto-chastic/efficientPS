import torch
import torch.nn as nn
from torchvision.ops import nms
import torch.utils.checkpoint as checkpoint

from . import *


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output


def outputSizeDeconv(in_size, kernel_size, stride, padding, output_padding):
    return int(
        (in_size - 1) * stride
        - 2 * padding
        + (kernel_size - 1)
        + output_padding
        + 1
    )


def convert_box_vertices_to_cwh(bbox):
    if not isinstance(bbox, torch.Tensor):
        bbox = torch.tensor(
            [
                bbox[0][0],
                bbox[0][1],
                bbox[1][0],
                bbox[1][1],
            ]
        )
    bbox = bbox.float()
    vt = torch.zeros_like(bbox)
    vt[..., 2] = bbox[..., 2] - bbox[..., 0]
    vt[..., 3] = bbox[..., 3] - bbox[..., 1]
    vt[..., 0] = bbox[..., 0] + vt[..., 2] / 2
    vt[..., 1] = bbox[..., 1] + vt[..., 3] / 2

    return vt


def convert_box_chw_to_vertices(bbox):
    vt = torch.zeros_like(bbox)
    vt[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    vt[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    vt[..., 2] = bbox[..., 0] + bbox[..., 2] / 2
    vt[..., 3] = bbox[..., 1] + bbox[..., 3] / 2
    return vt


class RegionProposalOutput:
    def __init__(self):
        self.scale = 0
        self.anchors = None  # List[(w*h*K)x4, ..., (w*h*K)x4]  len = batch
        self.objectness = None  # List[(w*h*K)x1, ..., (w*h*K)x1]  len = batch
        self.transformations = (
            None  # List[(w*h*K)x1, ..., (w*h*K)x1]  len = batch
        )

    def get_anch_obj(self):
        return self.anchors, self.objectness


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

    @staticmethod
    def checkpointer(function):
        def custom_forward(*inputs):
            inputs = function(inputs[0])
            return inputs
        return custom_forward

    def forward(self, x):
        x = checkpoint.checkpoint(
            self.checkpointer(self.forward_), x
        )
        return x

    def forward_(self, x):
        x = self.conv(x)
        x = self.depth_conv(x)
        return x


# def conv_3x3_bn(in_channels, out_channels, stride, activation=nn.LeakyReLU):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         activation(inplace=True)
#     )


def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn_custom_act(in_channels, out_channels, activation=nn.Sigmoid):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )


class MobileInvertedBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
        activation=nn.ReLU6,
    ):
        super(MobileInvertedBottleneck, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(in_channels * expand_ratio)
        self.residual_shortcut = (
            self.stride == 1 and in_channels == out_channels
        )

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
            # dw
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride,
                1,
                groups=hidden_dim,
                bias=False,
            ),
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


class DensePredictionCell(nn.Module):
    def __init__(self, activation=nn.LeakyReLU):
        super(DensePredictionCell, self).__init__()
        self.activation = activation()

        self.conv1 = DepthSeparableConv2d(
            256, 256, kernel_size=3, stride=1, padding=(1, 6), dilation=(1, 6)
        )
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = DepthSeparableConv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = DepthSeparableConv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=(6, 21),
            dilation=(6, 21),
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = DepthSeparableConv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=(18, 15),
            dilation=(18, 15),
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = DepthSeparableConv2d(
            256, 256, kernel_size=3, stride=1, padding=(6, 3), dilation=(6, 3)
        )
        self.bn5 = nn.BatchNorm2d(256)

        self.conv_final = conv_1x1_bn(1280, 128)

    def forward(self, x):
        x_1 = self.activation(self.bn1(self.conv1(x)))

        x_2 = self.activation(self.bn2(self.conv2(x_1)))
        x_3 = self.activation(self.bn3(self.conv3(x_1)))
        x_4 = self.activation(self.bn4(self.conv4(x_1)))
        x_5 = self.activation(self.bn5(self.conv5(x_4)))

        block = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=1)

        return self.conv_final(block)


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchors, scale, activation=nn.LeakyReLU):
        super(RegionProposalNetwork, self).__init__()
        self.activation = activation()
        self.anchors = (
            anchors / scale
        )  # Sample: torch.tensor([[0.0, 0.0, 22.0, 22.0]])
        # self.anchors = convert_box_chw_to_vertices(self.anchors)
        self.scale = scale

        self.num_anchors = len(anchors)

        self.conv1 = DepthSeparableConv2d(256, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.anchors_conv = conv_1x1_bn_custom_act(
            256, self.num_anchors * 4, activation=None
        )
        self.objectness_conv = conv_1x1_bn_custom_act(256, self.num_anchors)

    def forward(self, x):
        batch, height, width = x.shape[0], x.shape[2], x.shape[3]

        x = self.activation(self.bn1(self.conv1(x)))

        anchors_correction = self.anchors_conv(x)  # Bx(4k)xHxW
        objectness = self.objectness_conv(x)  # BxKxHxW

        anchors_batch = torch.stack(batch * [self.anchors]).view(
            batch, self.num_anchors, 4, 1, 1
        )  # Bx4xKx1
        anchors_correction = anchors_correction.view(
            batch, self.num_anchors, 4, height, width
        )

        x_bb_position = (
            torch.arange(0, height).view(1, 1, height, 1).to(x.device)
        )
        y_bb_position = (
            torch.arange(0, width).view(1, 1, 1, width).to(x.device)
        )

        transformations = torch.zeros_like(anchors_correction)

        transformations[:, :, 0, ...] = (
            transformations[:, :, 0, ...] + x_bb_position
        )
        transformations[:, :, 1, ...] = (
            transformations[:, :, 1, ...] + y_bb_position
        )

        anchors_correction += transformations

        transformations[:, :, 2:, ...] = (
            transformations[:, :, 2:, ...] + anchors_batch[:, :, 2:, ...]
        )

        corrected_position = (
            anchors_correction[:, :, :2, ...] + anchors_batch[:, :, :2, ...]
        )
        corrected_size = (
            torch.exp(anchors_correction[:, :, 2:, ...])
            * anchors_batch[:, :, 2:, ...]
        )

        corrected_anchors = torch.cat(
            (corrected_position, corrected_size), 2
        )  # BxKx4xWxH

        format_anchors = (
            corrected_anchors.permute((0, 2, 1, 3, 4))
            .reshape(batch, 4, -1)
            .permute((0, 2, 1))
        )

        if torch.sum(format_anchors.isnan()).item() > 0:
            format_anchors

        objectness = objectness.view(batch, self.num_anchors, 1, -1)
        format_objectness = (
            objectness.permute((0, 2, 1, 3))
            .reshape(batch, 1, -1)
            .permute((0, 2, 1))
        )

        transformations = (
            transformations.permute((0, 2, 1, 3, 4))
            .reshape(batch, 4, -1)
            .permute((0, 2, 1))
        )

        list_anchors = [
            x.squeeze() for x in torch.chunk(format_anchors, batch)
        ]  # List[(w*h*K)x4, ..., (w*h*K)x4]  len = batch

        list_objectness = [
            x.squeeze() for x in torch.chunk(format_objectness, batch)
        ]  # List[(w*h*K)x1, ..., (w*h*K)x1]  len = batch

        list_transformations = [
            x.squeeze() for x in torch.chunk(transformations, batch)
        ]  # List[(w*h*K)x1, ..., (w*h*K)x1]  len = batch

        output = RegionProposalOutput()

        output.scale = self.scale
        output.anchors = list_anchors
        output.objectness = list_objectness
        output.transformations = list_transformations
        return output


if __name__ == "__main__":
    # dpc = DensePredictionCell()
    # out = dpc(torch.rand(3, 256, 32, 64))
    # print("out", out.shape)
    rpn = RegionProposalNetwork(ANCHORS, 32)
    out = rpn(torch.rand(3, 256, 32, 64))
    print(out.anchors[0].shape)
