import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import (
    DepthSeparableConv2d,
    MobileInvertedBottleneck,
    conv_1x1_bn,
)
from .fpn import TwoWayFeaturePyramid
from .ss_head import SemanticSegmentationHead
from .is_head import InstanceSegmentationHead


class FullModel(nn.Module):
    def __init__(self, num_things, num_stuff, anchors, nms_threshold, activation=nn.LeakyReLU):
        super(FullModel, self).__init__()

        self.fpn = TwoWayFeaturePyramid(activation)
        self.ss_head = SemanticSegmentationHead(num_stuff+num_things, activation)
        self.is_head = InstanceSegmentationHead(num_things, anchors, nms_threshold,activation)

    def forward(self, inp):
        # Main and bottom-up
        p32, p16, p8, p4 = self.fpn(inp)
        semantic_logits = self.ss_head(p32, p16, p8, p4)
        classes, bboxes, mask_logits = self.is_head(p32, p16, p8, p4)

        return semantic_logits, classes, bboxes, mask_logits


if __name__ == "__main__":
    anchors = torch.tensor([[1.0, 1.0, 220.0, 320.0], [1.0, 1.0, 320.0, 220.0]]).cuda()

    full = FullModel(10, 8, anchors, 0.3).cuda()
    semantic_logits, classes, bboxes, mask_logits = full(torch.rand(1, 3, 512, 1024).cuda())
    print("semantic_logits, classes, bboxes, mask", semantic_logits.shape, classes.shape, bboxes.shape, mask_logits.shape)

