import torch
import torch.nn as nn
import torch.nn.functional as F
from . import *
from ..train_core.utilities import index_select2D


from .utilities import (
    DepthSeparableConv2d,
    MobileInvertedBottleneck,
    conv_1x1_bn,
)
from .fpn_pretrained import TwoWayFeaturePyramid
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

    def get_classes(self):
        return torch.exp(self.classes[:, 1:, :]) / torch.sum(torch.exp(self.classes[:, 1:, :]), 1)

    def get_confidence(self):
        return 1 - torch.exp(self.classes[:, 0, :])

    def pick_mask_class_conf_bboxes(self):
        index = torch.argmax(self.get_classes(), 1)
        
        mask_logits = []
        confidences = []
        bboxes = []
        for b in range(self.bboxes.shape[0]):
            mask_logits.append(
                index_select2D(
                    self.mask_logits[b].permute(2, 3, 1, 0),
                    index[b]
                )
            )
            bboxes.append(
                index_select2D(
                    self.bboxes[b].permute(1, 0, 2),
                    index[b]
                )
            )
            confidences.append(
                self.get_confidence()[b]
            )

        mask_logits = torch.stack(mask_logits)
        bboxes = torch.stack(bboxes)
        conficence = torch.stack(confidences)

        classes = index + 1
        return mask_logits, classes, conficence, bboxes



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
            num_things + num_stuff, activation
        )
        self.is_head = InstanceSegmentationHead(
            num_things, anchors, nms_threshold, activation
        )
        self.apply(self._initialize_weights)

    def forward(self, inp):
        # Main and bottom-up
        p32, p16, p8, p4 = self.fpn(inp)      
        print("state: fpn")
        semantic_logits = self.ss_head(p32, p16, p8, p4)
        print("state: ss")
        (
            classes,
            bboxes,
            mask_logits,
            proposed_bboxes,
            primitive_anchors,
        ) = self.is_head(p32, p16, p8, p4)
        print("state: is")
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
    anchors = ANCHORS.cuda()

    full = FullModel(10, 8, anchors, 0.3).cuda()
    out = full(torch.rand(1, 3, 512, 1024).cuda())
