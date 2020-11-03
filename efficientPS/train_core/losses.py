import torch
import torch.nn as nn
from .utilities import iou_function
from ..models.utilities import (
    convert_box_chw_to_vertices,
    convert_box_vertices_to_cwh,
)
from ..models.full import ANCHORS


class LossFunctions:
    def __init__(self, ground_truth, inference):
        self.ground_truth = ground_truth
        self.inference = inference

        self.objectness_thr = 0.5  # According to Mask R-CNN paper

        self.first_stage_num_samples = 256  # According to EfficientPS paper

    def ss_loss_max_pooling(self):
        logits = self.inference.semantic_logits
        (batch, num_classes, height, width) = logits.shape
        nlll = nn.NLLLoss(reduction="none")
        loss = nlll(logits, self.ground_truth.get_label_IDs())

        values, ind = torch.sort(loss.view(batch, -1), 1)

        middle = int(val.shape[1] / 2)
        top_quartile = int(middle + middle / 2)

        top_quartile_values = val[:, top_quartile]
        mask = (
            loss.ge(top_quartile_values.unsqueeze(-1).unsqueeze(-1))
            * 4
            / height
            / width
        )

        return torch.sum(loss * mask) / batch

    def objectness_loss(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()

        # How many samples, positive and negative, per bbox we sample
        num_samples_per_box_stage = (
            self.first_stage_num_samples // len(gt_bb) // 4
        )

        objectness_loss = 0
        regression_loss = 0
        for bb in gt_bb:
            for i in range(4):
                level_primitives = self.primitive_anchors[i]
                primitive_bb = level_primitives.anchors
                primitive_obj = level_primitives.objectness
                primitive_transformations = level_primitives.transformations

                samples = random.sample(
                    range(primitive_bb[0].shape[0]), num_samples_per_box_stage
                )

                edges_bb = convert_box_chw_to_vertices(primitive_bb[0])
                gt_objectness = iou_function(edges_bb, bb["box"]).ge(
                    self.objectness_thr
                )
                partial_objectness_loss = gt_objectness * torch.log(
                    primitive_obj
                ) + (1 - gt_objectness) * torch.log(1 - primitive_obj)
                objectness_loss += partial_objectness_loss[samples]

                positive_indeces = gt_objectness[:, 0].nonzero()

                regression_loss += self._object_proposal_loss(
                    primitive_bb[0],
                    primitive_transformations[0],
                    bb["box"],
                    positive_indeces,
                )

        return (
            objectness_loss / num_samples_per_box_stage * 4,
            regression_loss / num_samples_per_box_stage * 4,
        )

    def _object_proposal_loss(
        self, anchors_bb, transformations, bb, positive_indeces
    ):
        def bb_parametrization(anchors):
            param = torch.zeros_like(anchors)
            param[:, :2] = (
                anchors[:, :2] - transformations[:, :2]
            ) / transformations[:, 2:]
            param[:, 2:] = torch.log(anchors[:, 2:] / transformations[:, 2:])
            return param

        param_anchors = bb_parametrization(
            convert_box_vertices_to_cwh(anchors_bb)
        )

        bb_tensor = torch.zeros_like(anchors_bb) + convert_box_vertices_to_cwh(
            bb
        ).to(anchors_bb.device)
        param_bb = bb_parametrization(bb_tensor)

        l1 = nn.SmoothL1Loss(reduction="none")  # According to Fast R-CNN paper
        loss = l1(param_anchors, param_bb)

        return torch.sum(loss.index_select(0, positive_indeces)), torch.shape(
            positive_indeces
        )


if __name__ == "__main__":
    pass
