import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign

from .utilities import DepthSeparableConv2d, RegionProposalNetwork


class InstanceSegmentationHead(nn.Module):
    def __init__(self, num_anchors, activation=nn.LeakyReLU):
        super(InstanceSegmentationHead, self).__init__()

        self.rps_p32 = RegionProposalNetwork()
        self.rps_p16 = RegionProposalNetwork()
        self.rps_p8 = RegionProposalNetwork()
        self.rps_p4 = RegionProposalNetwork()

        self.roi_align_32 = RoIAlign((14, 14), spatial_scale=32)
        self.roi_align_16 = RoIAlign((14, 14), spatial_scale=16)
        self.roi_align_8 = RoIAlign((14, 14), spatial_scale=8)
        self.roi_align_4 = RoIAlign((14, 14), spatial_scale=4)

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
