import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign

from .utilities import DepthSeparableConv2d, RegionProposalNetwork


class InstanceSegmentationHead(nn.Module):
    def __init__(self, anchors, activation=nn.LeakyReLU):
        super(InstanceSegmentationHead, self).__init__()
        # An anchor is centered at the sliding windowin question,
        # and is associated with a scale and aspectratio.
        # By  default  we  use  3  scales  and3 aspect ratios, yielding
        # k= 9anchors at each slidingposition.  For  a  convolutional  
        # feature  map  of  a  sizeW×H(typically∼2,400), there are WHk anchors intotal.
        self.rps_p32 = RegionProposalNetwork(anchors/32)
        self.rps_p16 = RegionProposalNetwork(anchors/16)
        self.rps_p8 = RegionProposalNetwork(anchors/8)
        self.rps_p4 = RegionProposalNetwork(anchors/4)

        self.roi_align = RoIAlign((14, 14), spatial_scale=32)

    def forward(self, p32, p16, p8, p4):
        # Main and bottom-up
        p32_anchors, p32_objectness = self.rps_p32(p32)
        p16_anchors, p16_objectness = self.rps_p16(p16)
        p8_anchors, p8_objectness = self.rps_p8(p8)
        p4_anchors, p4_objectness = self.rps_p4(p4)

        # these are supposed to be a list wont work, change
        p32_N = torch.max(torch.ones_like(p32_anchors[:, 0]), torch.min(torch.ones_like(p32_anchors[:, 0])*4, torch.floor(3 + torch.log2(torch.sqrt(p32_anchors[:,0]*p32_anchors[:,1] * 32**2)/224)))) 
        p16_N = torch.max(torch.ones_like(p16_anchors[:, 0]), torch.min(torch.ones_like(p16_anchors[:, 0])*4, torch.floor(3 + torch.log2(torch.sqrt(p16_anchors[:,0]*p16_anchors[:,1] * 16**2)/224)))) 
        p8_N = torch.max(torch.ones_like(p8_anchors[:, 0]), torch.min(torch.ones_like(p8_anchors[:, 0])*4, torch.floor(3 + torch.log2(torch.sqrt(p8_anchors[:,0]*p8_anchors[:,1] * 8**2)/224)))) 
        p4_N = torch.max(torch.ones_like(p4_anchors[:, 0]), torch.min(torch.ones_like(p4_anchors[:, 0])*4, torch.floor(3 + torch.log2(torch.sqrt(p4_anchors[:,0]*p4_anchors[:,1] * 4**2)/224)))) 

        return p32, p16, p8, p4


if __name__ == "__main__":
    fpn = TwoWayFeaturePyramid().cuda()
    p32, p16, p8, p4 = fpn(torch.rand(3, 3, 1024, 2048))
    print("p32", p32.shape)
    print("p16", p16.shape)
    print("p8", p8.shape)
    print("p4", p4.shape)
