import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import DepthSeparableConv2d, MobileInvertedBottleneck, conv_1x1_bn
from .fpn import TwoWayFeaturePyramid
from .ss_head import SemanticSegmentationHead

class FullModel(nn.Module):
    def __init__(self, num_classes, activation=nn.LeakyReLU):
        super(FullModel, self).__init__()

        self.fpn = TwoWayFeaturePyramid()
        self.ss_head = SemanticSegmentationHead(num_classes)


    def forward(self, inp):
        # Main and bottom-up

        return p32, p16, p8, p4


if __name__ == "__main__":
    fpn = TwoWayFeaturePyramid().cuda()
    p32, p16, p8, p4 = fpn(torch.rand(3, 3, 1024, 2048))
    print("p32", p32.shape)
    print("p16", p16.shape)
    print("p8", p8.shape)
    print("p4", p4.shape)