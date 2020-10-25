import torch
import torch.nn as nn
from torchvision.ops import nms

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
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


def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def conv_1x1_bn_sig(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid(),
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
    def __init__(self, anchors, nms_threshold=0.9, activation=nn.LeakyReLU):
        super(RegionProposalNetwork, self).__init__()
        self.activation = activation()
        self.nms_threshold = nms_threshold
        self.anchors = anchors  # torch.tensor([[-2.0, -2.0, 22.0, 22.0], [-2.0, -2.0, 22.0, 22.0]])
        # torch.tensor([[0, -2.0, -2.0, 22.0, 22.0], [0, -2.0, -2.0, 22.0, 22.0]])
        # First value corresponds to which image in the batch the anchor is applied to
        self.num_anchors = len(anchors)

        self.conv1 = DepthSeparableConv2d(256, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.anchors_conv = conv_1x1_bn(256, self.num_anchors * 4)
        self.objectness_conv = conv_1x1_bn_sig(256, self.num_anchors)

    def forward(self, x):
        batch = x.shape[0]
        x = self.activation(self.bn1(self.conv1(x)))

        anchors_correction = self.anchors_conv(x)  # Bx(4k)xHxW
        objectness = self.objectness_conv(x) # BxKxHxW

        anchors_batch = torch.stack(batch * [self.anchors]).view(
            batch, self.num_anchors, 4, 1
        )  # Bx4xKx1
        anchors_correction = anchors_correction.view(batch, self.num_anchors, 4, -1)

        corrected_position = (
            anchors_correction[:, :, :2, :] + anchors_batch[:, :, :2, :]
        )
        corrected_size = (
            torch.exp(anchors_correction[:, :, 2:, :])
            * anchors_batch[:, :, 2:, :]
        )

        corrected_anchors = torch.cat(
            (corrected_position, corrected_size), 2
        )  # BxKx4x(w*h)
        format_anchors = (
            corrected_anchors.permute((0, 2, 1, 3))
            .reshape(batch, 4, -1)
            .permute((0, 2, 1))
        )

        objectness = objectness.view(batch, self.num_anchors, 1, -1)
        format_objectness = (
            objectness.permute((0, 2, 1, 3))
            .reshape(batch, 1, -1)
            .permute((0, 2, 1))
        )

        nms_indices = nms(
            format_anchors.squeeze_(),
            format_objectness.squeeze_(),
            iou_threshold=self.nms_threshold
        )

        format_anchors = format_anchors.index_select(0, nms_indices)
        format_objectness = format_objectness.index_select(0, nms_indices)

        list_anchors = [
            x.squeeze() for x in torch.chunk(format_anchors, batch)
        ]  # List[(w*h*K)x4, ..., (w*h*K)x4]  len = batch

        list_objectness = [
            x.squeeze() for x in torch.chunk(format_objectness, batch)
        ]  # List[(w*h*K)x1, ..., (w*h*K)x1]  len = batch

        return list_anchors, list_objectness


if __name__ == "__main__":
    dpc = DensePredictionCell()
    out = dpc(torch.rand(3, 256, 32, 64))
    print("out", out.shape)
