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


class PSOutput:
    def __init__(
        self,
        semantic_logits,
        classes,
        bboxes,
        mask_logits,
        proposed_bboxes,
        primitive_anchors,
    ):
        self.semantic_logits = semantic_logits
        self.classes = classes
        self.bboxes = bboxes
        self.mask_logits = mask_logits
        self.proposed_bboxes = proposed_bboxes
        self.primitive_anchors = primitive_anchors


class FullModel(nn.Module):
    def __init__(
        self,
        num_things,
        num_stuff,
        anchors,
        nms_threshold,
        activation=nn.LeakyReLU,
    ):
        super(FullModel, self).__init__()

        self.fpn = TwoWayFeaturePyramid(activation)
        self.ss_head = SemanticSegmentationHead(
            num_stuff + num_things, activation
        )
        self.is_head = InstanceSegmentationHead(
            num_things, anchors, nms_threshold, activation
        )
        self.apply(self._initialize_weights)

    def forward(self, inp):
        # Main and bottom-up
        p32, p16, p8, p4 = self.fpn(inp)
        semantic_logits = self.ss_head(p32, p16, p8, p4)
        (
            classes,
            bboxes,
            mask_logits,
            proposed_bboxes,
            primitive_anchors,
        ) = self.is_head(p32, p16, p8, p4)

        return PSOutput(
            semantic_logits,
            classes,
            bboxes,
            mask_logits,
            proposed_bboxes,
            primitive_anchors,
        )

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain("leaky_relu")
            )
            # nn.init.constant_(m.bias, 0.1)

    def save_model(self, path, name="EPSFull"):
        if not path.endswith("/"):
            path = "{}/".format(path)
        self.save("{}{}.pt".format(path, name))

    def load_model(self, path, name="EPSFull"):
        if not path.endswith("/"):
            path = "{}/".format(path)
        self.load("{}{}.pt".format(path, name))


if __name__ == "__main__":
    anchors = torch.tensor(
        [[1.0, 1.0, 220.0, 320.0], [1.0, 1.0, 320.0, 220.0]]
    ).cuda()

    full = FullModel(10, 8, anchors, 0.3).cuda()
    out = full(torch.rand(1, 3, 512, 1024).cuda())
    # print(
    #     "semantic_logits, classes, bboxes, mask",
    #     semantic_logits.shape,
    #     classes.shape,
    #     bboxes.shape,
    #     mask_logits.shape,
    # )
